from collections import OrderedDict

import torch
import numpy as np


class FilterParams(torch.nn.Module):
    def __init__(self, params, name_shape_dict):
        super(FilterParams, self).__init__()

        self.params = params
        self.filter_params = OrderedDict()
        self.filter_params_t = OrderedDict()

        self._init_filter_params(name_shape_dict)

    def _init_filter_params(self, name_shape_dict):
        for name, init_value in name_shape_dict.items():
            if hasattr(init_value, '__call__'):
                self.filter_params[name] = init_value
            elif type(init_value) is np.ndarray:
                self.filter_params[name] = torch.nn.Parameter(torch.tensor(init_value, dtype=self.params.torch_d_type), requires_grad=True)
            else:
                self.filter_params[name] = torch.nn.Parameter(torch.tensor(init_value, dtype=self.params.torch_d_type), requires_grad=True)

    def update_with_transform(self, transform, name):
        self.filter_params[name].data = transform.inv_transform(self.filter_params[name].data)
        self.filter_params_t[name] = lambda: transform.transform(self.filter_params[name])

    def update_param(self, param_name, value):
        if hasattr(value, '__call__'):
            self.filter_params[param_name] = value
        elif type(value) is np.ndarray:
            self.filter_params[param_name].data = torch.tensor(value, requires_grad=True, dtype=self.params.torch_d_type)
        else:
            self.filter_params_torch[param_name].data = torch.tensor(np.array([value]), requires_grad=True, dtype=self.params.torch_d_type)


