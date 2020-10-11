from collections import OrderedDict
from GLM.GLM_Model.BoundTransform import BoundTransform

import torch
import numpy as np


class GpParams:
    def __init__(self, params, name_shape_dict):
        self.params = params
        self.gp_params = OrderedDict()
        self.gp_params_t = OrderedDict()
        self.gp_params_it = OrderedDict()
        self.gp_kernel = OrderedDict()

        self.transform_obj = OrderedDict()
        self._init_gp_params(name_shape_dict)

    def _init_gp_params(self, name_shape_dict):
        for name, init_val_grad in name_shape_dict.items():
            init_value = init_val_grad[0]
            requires_grad = init_val_grad[1]

            if hasattr(init_value, '__call__'):
                self.gp_kernel[name] = init_value
                self.gp_kernel[name] = init_value
            elif type(init_value) is np.ndarray:
                self.gp_params[name] = torch.nn.Parameter(torch.tensor(init_value, dtype=self.params.torch_d_type), requires_grad=requires_grad)
            else:
                self.gp_params[name] = torch.nn.Parameter(torch.tensor(np.array([init_value]), dtype=self.params.torch_d_type), requires_grad=requires_grad)

    def update_with_transform(self, transform, name):
        self.transform_obj[name] = transform
        self.gp_params[name].data = transform.inv_transform(self.gp_params[name].data)
        self.gp_params_t[name] = lambda: transform.transform(self.gp_params[name])
        self.gp_params_it[name] = transform.inv_transform

    def update_with_transform_override_bounds(self, new_val, name):
        lower_bound = min(self.transform_obj[name].bounds[0], new_val)
        upper_bound = max(self.transform_obj[name].bounds[1], new_val)

        self.transform_obj[name] = BoundTransform(self.params, [lower_bound, upper_bound])

        if isinstance(new_val, np.ndarray):
            self.gp_params[name].data = self.transform_obj[name].inv_transform(torch.tensor(new_val), dtype=self.params.torch_d_type)
        else:
            self.gp_params[name].data = self.transform_obj[name].inv_transform(torch.tensor(np.array([new_val]), dtype=self.params.torch_d_type))

        self.gp_params_t[name] = lambda: self.transform_obj[name].transform(self.gp_params[name])
        self.gp_params_it[name] = self.transform_obj[name].inv_transform

    def update_with_new_bounds(self, name, hard_bounds):
        lowest_value = torch.min(self.gp_params_t[name]().data).data.detach().numpy()
        highest_value = torch.max(self.gp_params_t[name]().data).data.detach().numpy()

        lower_bound = (lowest_value - self.params.gp_hyperparameter_percent_bound * lowest_value
                       if
                       lowest_value - self.params.gp_hyperparameter_percent_bound * lowest_value > hard_bounds[0]
                       else
                       hard_bounds[0])

        upper_bound = (highest_value + self.params.gp_hyperparameter_percent_bound * highest_value
                       if
                       highest_value + self.params.gp_hyperparameter_percent_bound * highest_value < hard_bounds[1]
                       else
                       hard_bounds[1])

        current_data_post_transform = self.gp_params_t[name]().data.clone()
        self.transform_obj[name] = BoundTransform(self.params, [lower_bound, upper_bound])
        self.gp_params[name].data = self.transform_obj[name].inv_transform(current_data_post_transform)
        self.gp_params_t[name] = lambda: self.transform_obj[name].transform(self.gp_params[name])
        self.gp_params_it[name] = self.transform_obj[name].inv_transform


    def update_param(self, param_name, value):
        if hasattr(value, '__call__'):
            self.gp_kernel[param_name] = value
            self.gp_kernel[param_name] = value
        elif type(value) is np.ndarray:
            self.gp_params_numpy[param_name] = value
            self.gp_params_torch[param_name].data = torch.tensor(value, requires_grad=True, dtype=torch.double)
        else:
            self.gp_params_numpy[param_name] = np.array([value])
            self.gp_params_torch[param_name].data = torch.tensor(np.array([value]), requires_grad=True, dtype=torch.double)


