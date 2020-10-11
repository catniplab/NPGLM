import torch
import numpy as np

class BoundTransform:
    def __init__(self, params, bounds):
        self.params = params
        self.bounds = bounds
        self.transform = None
        self.inv_transform = None

        if torch.is_tensor(self.bounds[0]) and bounds[0].shape[0] > 1:
            self._get_full_box_transform()

        elif bounds[0] != -np.inf and bounds[1] == np.inf:
            self._get_half_box_transform()

        elif bounds[0] == -np.inf and bounds[1] != np.inf:
            self._get_half_box_transform()

        elif bounds[0] != np.inf and bounds[1] != np.inf:
            self._get_full_box_transform()

        elif bounds[0] == -np.inf and bounds[1] == np.inf:
            self._get_identity_transform()

    def _get_half_box_transform(self):
        if self.bounds[1] == np.inf:
            self.transform = lambda x: torch.nn.Softplus(beta=1, threshold=20)(x) + self.bounds[0]
            self.inv_transform = lambda x: 1 * torch.log(torch.exp(x - self.bounds[0]) - 1 + 1e-12)
        elif self.bounds[0] == -np.inf:
            self.transform = lambda x: -1 * torch.nn.Softplus(beta=1, threshold=20)(x) + self.bounds[1]
            self.inv_transform = lambda x: 1 * torch.log(torch.exp(-1 * (x - self.bounds[1])))

    def _get_full_box_transform(self):
        atanh = lambda x: 0.5 * (torch.log(1+x) - torch.log(1-x))
        bound_length = self.bounds[1] - self.bounds[0]
        bound_half_length = bound_length / 2

        self.transform = lambda x: self.bounds[0] + bound_half_length * (torch.tanh(x/bound_half_length) + 1)
        self.inv_transform = lambda x: atanh((x - self.bounds[0]) / bound_half_length - 1) * bound_half_length

    def _get_identity_transform(self):
        self.transform = lambda x: x
        self.inv_transform = lambda x: x

