import numpy as np
import torch

from collections import OrderedDict
from GLM.GLM_Model.BoundTransform import BoundTransform

class TimeTracker:
    def __init__(self, params, **kwargs):
        self.params = params
        self.delta = params.delta

        self.offset = kwargs['offset_init']
        self.duration = kwargs['duration_init']
        self.time_plot_min = kwargs['time_plot_min']
        self.time_plot_max = kwargs['time_plot_max']
        self.use_every_blank_points = kwargs['use_every_blank_points']
        self.is_for_hist = kwargs['is_hist']

        self.triu_dx = None
        self.time_plot = None
        self.num_inducing = None
        self.init_time_u_dx = None

        self.time_dict = OrderedDict()
        self.time_dict_t = OrderedDict()

        self.hard_time_bounds = None
        self.bounds_transform = None
        self.current_time_bounds = None

        self._init_tracker()

    def _init_tracker(self):
        self.time_plot = torch.tensor(self.delta * np.arange(self.time_plot_min, self.time_plot_max + 1e-12), dtype=self.params.torch_d_type)

        time_x = self.delta * np.arange(self.offset, self.offset + self.duration)
        all_dx = np.arange(time_x.shape[0])
        self.time_dict['x'] = torch.tensor(time_x, dtype=self.params.torch_d_type)

        if self.is_for_hist:
            n = self.time_dict['x'].shape[0] // self.use_every_blank_points
            self.init_time_u_dx = self.gen_log_space(time_x.shape[0], n)
        else:
            self.init_time_u_dx = all_dx[0::self.use_every_blank_points]

        offset = -1 * self.params.delta/1.5 if self.is_for_hist else self.params.delta/2
        time_u = time_x[self.init_time_u_dx] + offset
        self.time_dict['u'] = torch.nn.Parameter(torch.tensor(time_u, dtype=self.params.torch_d_type), requires_grad=True)

        self.num_inducing = self.time_dict['u'].shape[0]
        self.triu_dx = np.triu_indices(self.num_inducing)

    def add_design_matrix_points(self, new_times):
        updated_design_matrix_times = np.sort(np.concatenate([self.time_dict['x'].data.detach().numpy(), new_times]))
        self.time_dict['x'].data = torch.tensor(updated_design_matrix_times, dtype=self.params.torch_d_type)

    def initialize_transform(self, hard_bounds, name):
        self.hard_time_bounds = hard_bounds
        self.update_current_bounds(self.time_dict[name].data)
        self.bounds_transform = BoundTransform(self.params, self.current_time_bounds)
        self.time_dict[name].data = self.bounds_transform.inv_transform(self.time_dict[name].data)
        self.time_dict_t[name] = lambda: self.bounds_transform.transform(self.time_dict[name])

    def update_current_bounds(self, time_u):
        current_lower_bound = (time_u.min().data.detach().numpy() - self.params.time_optimization_radius
                               if (time_u.min().data.detach().numpy() - self.params.time_optimization_radius) > self.hard_time_bounds[0]
                               else self.hard_time_bounds[0])
        current_upper_bound = (time_u.max().data.detach().numpy() + self.params.time_optimization_radius
                               if (time_u.max().data.detach().numpy() + self.params.time_optimization_radius) < self.hard_time_bounds[1]
                               else self.hard_time_bounds[1])
        self.current_time_bounds = [current_lower_bound, current_upper_bound]

    def update_with_new_bounds(self, name):
        lower_bound = torch.clamp(self.time_dict_t[name]().data.clone() - self.params.time_optimization_radius,
                                  min=self.hard_time_bounds[0])
        upper_bound = torch.clamp(self.time_dict_t[name]().data.clone() + self.params.time_optimization_radius,
                                  max=self.hard_time_bounds[1])

        current_data_post_transform = self.time_dict_t[name]().data.clone()
        self.bounds_transform = BoundTransform(self.params, [lower_bound, upper_bound])
        self.time_dict[name].data = self.bounds_transform.inv_transform(current_data_post_transform)
        self.time_dict_t[name] = lambda: self.bounds_transform.transform(self.time_dict[name])

    def update_transform(self, name):
        self.update_current_bounds(self.time_dict_t[name].data)
        self.bounds_transform = BoundTransform(self.params, self.current_time_bounds)
        self.time_dict_t[name] = lambda: self.bounds_transform.transform(self.time_dict[name])

    def gen_log_space(self, limit, n):
        result = [1]
        if n > 1:  # just a check to avoid ZeroDivisionError
            ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
        while len(result) < n:
            next_value = result[-1] * ratio
            if next_value - result[-1] >= 1:
                # safe zone. next_value will be a different integer
                result.append(next_value)
            else:
                # problem! same integer. we need to find next_value by artificially incrementing previous value
                result.append(result[-1] + 1)
                # recalculate the ratio so that the remaining values will scale correctly
                ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
        # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
        return np.array(list(map(lambda x: round(x) - 1, result)), dtype=np.uint64)




