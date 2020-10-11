import torch
import numpy as np
from functools import reduce
from collections import OrderedDict

class PyTorchObjective(object):
    """PyTorch objective function, wrapped to be called by scipy.optimize."""
    def __init__(self, obj_module, parameters_to_opt, scipy_closure):
        self.f = obj_module # some pytorch module, that produces a scalar loss
        self.parameters_to_opt = parameters_to_opt
        self.scipy_closure = scipy_closure
        # make an x0 from the parameters in this module
        parameters = OrderedDict()
        for param_name, param in obj_module.named_parameters():
            if param_name in parameters_to_opt:
                parameters[param_name] = param
        self.param_shapes = {n:parameters[n].size() for n in parameters}
        # ravel and concatenate all parameters to make x0
        self.x0 = np.concatenate([parameters[n].data.numpy().ravel()
                                   for n in parameters])

    def unpack_parameters(self, x):
        """optimize.minimize will supply 1D array, chop it up for each parameter."""
        i = 0
        named_parameters = OrderedDict()
        for n in self.param_shapes:
            param_len = reduce(lambda x,y: x*y, self.param_shapes[n])
            # slice out a section of this length
            param = x[i:i+param_len]
            # reshape according to this size, and cast to torch
            param = param.reshape(*self.param_shapes[n])
            named_parameters[n] = torch.from_numpy(param)
            # update index
            i += param_len
        return named_parameters

    def pack_grads(self):
        """pack all the gradients from the parameters in the module into a
        numpy array."""
        grads = []
        for name, p in self.f.named_parameters():
            if name in self.parameters_to_opt:
                grad = p.grad.data.numpy()
                grads.append(grad.ravel())
        return np.concatenate(grads)

    def is_new(self, x):
        # if this is the first thing we've seen
        if not hasattr(self, 'cached_x'):
            return True
        else:
            # compare x to cached_x to determine if we've been given a new input
            x, self.cached_x = np.array(x), np.array(self.cached_x)
            error = np.abs(x - self.cached_x)
            return error.max() > 1e-8

    def cache(self, x):
        # unpack x and load into module
        state_dict = self.unpack_parameters(x)

        for name, p in self.f.named_parameters():
            if name in self.parameters_to_opt:
                p.data = state_dict[name]
        # self.f.load_state_dict(state_dict)
        # store the raw array as well
        self.cached_x = x
        # zero the gradient
        # use it to calculate the objective
        obj = self.scipy_closure()
        # backprop the objective
        obj.backward()
        self.cached_f = obj.item()
        self.cached_jac = self.pack_grads()

    def fun(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.cached_f

    def jac(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.cached_jac