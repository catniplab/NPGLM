import torch
import numpy as np
import matplotlib.pyplot as plt
import NPGLM_v2p0.utils.utils as utils


class Covariate(torch.nn.Module):

    def __init__(self, name, delta, offset, length):
        super(Covariate, self).__init__()
        self.name = name
        self.delta = delta

        self.X = None
        self.Z = None
        self.U = None

        # filter parameters
        self.time_total = length
        self.time_offset = offset

        # variational parameters
        self.variational_mean = None
        self.variational_covariance_tril = None

        # gp parameters
        self.kernel_fn = None
        self.inducing_pt_loc = None
        self.num_inducing_pts = None

    def init_variational_parameters(self, std_glm_init=False):
        if not std_glm_init:
            num_tril_elements = (self.num_inducing_pts * (self.num_inducing_pts + 1)) // 2
            self.variational_mean = torch.nn.Parameter(torch.randn(self.num_inducing_pts))
            self.variational_covariance_tril = torch.nn.Parameter(torch.randn(num_tril_elements))
        else:
            # TODO
            pass

    def create_design_matrices(self):
        pass

