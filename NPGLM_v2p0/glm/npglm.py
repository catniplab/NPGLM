from collections import OrderedDict
import NPGLM_v2p0.utils.utils as utils
import torch


class NPGLM(torch.nn.Module):
    def __init__(self):
        self.torch_dtype = torch.float64
        self.Y = None
        self.b = None

        self.covariates = OrderedDict()

    def add_covariate(self, covariate):
        self.covariates[covariate.name] = covariate

    def init_npglm(self):
        for name, cov in self.covariates.items():
            cov.init_variational_parameters()

    def forward(self, X, Y):
        x_star_h = torch.zeros(Y.shape[1], dtype=self.torch_dtype)

        for k, covariate in self.covariates.items():
            x_star_h += utils.torch_batch_convolve(X[k], covariate.filter_mean, covariate.filter_offset)

        loss = Y @ x_star_h - torch.sum(torch.exp(x_star_h), dim=1)

    def sample_psth(self):
        pass
