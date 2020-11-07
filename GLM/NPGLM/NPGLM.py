from GLM.GLM_Model.Model_Runner import Model_Runner
from GLM.GLM_Model import GLM_Model_GP, GLM_Model_MAP, GP_Covariate, MAP_Covariate
from GLM.GLM_Model.PoissonVariational import PoissonVariational

from Utils import utils, SpikeGen
from collections import OrderedDict

import torch
import pandas as pd


class NPGLM:
    def __init__(self, expt_label, params_path, torch_d_type=torch.float64):
        self.expt = expt_label
        self.params_path = params_path
        self.torch_d_type = torch_d_type
        self.data_df = None
        self.params = None

        self.covariates = OrderedDict()
        self.bounds_params = OrderedDict()
        self.gp_params = OrderedDict()
        self.time_params = OrderedDict()

    def initialize_parameters(self):
        self.params = utils.Params('GLM/GLM_Params/params.json')
        self.params.torch_d_type = self.torch_d_type
        self.params.gp_filter_plot_path = f"Results_Data/{self.expt}/data/gp_filter_plot"
        self.params.basis_filter_plot_path = f"Results_Data/{self.expt}/data/basis_filter_plot"
        self.params.gp_ev_path = f"Results_Data/{self.expt}/data/neuron_gp_ev"
        self.params.basis_ev_path = f"Results_Data/{self.expt}/data/neuron_map_ev"
        self.data_df = pd.read_pickle(self.params.expt_problem_data_path)

    def run_initialization_scheme(self):
        self.params.inference_type = 'basis'
        self.gp = Model_Runner(self.params)
        self.gp.initialize_design_matrices()
        self.gp.create_map_covariates()
        self.gp.train_map()

    def run_initilization_npglm(self):
        self.params.inference_type = 'gp'
        self.gp = Model_Runner(self.params)
        self.gp.initialize_design_matrices_demo(self.data_df)
        self.gp.create_variational_covariates_demo()

    def add_covariate(self, name, bounds_p, gp_p, time_p, etc_p):
        cov = GP_Covariate.GP_Covariate(self.params, etc_p, self.gp.covariate_data[name],
                                         name=name,
                                         use_bases=False)

        cov.add_bounds_params(bounds_p)
        cov.add_gp_params(gp_p)
        cov.add_time_init(time_p)
        self.gp.add_covariate(cov)

    def train_npglm(self):
        self.gp.train_demo()