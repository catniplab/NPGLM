import numpy as np
import pandas as pd
from Utils import utils
from collections import OrderedDict


class PoissonMAP:
    def __init__(self, params, data_df, glm, use_callbacks=False):
        self.params = params
        self.delta = params.delta
        self.glm = glm
        self.data_df = data_df
        self.time_plot = None

        self.evolution_df_dict = OrderedDict()
        self.optimizer_params = None
        self.use_callbacks = use_callbacks

    def initialize_model(self):
        self.timer_obj = utils.ResumableTimer()
        self.timer_obj.start()
        self.glm.initialize_covariates()

        for name, cov_s in self.data_df.iterrows():
            data_df_cols = ['plot_time', 'plot_2std', 'plot_mean',
                            'entire_time', 'entire_2std', 'entire_mean', 'entire_basis', 'entire_coeff', 'entire_ard_coeff', 'entire_cov',
                            'baseline', 'nll', 'nll_test', 'time_so_far']
            self.evolution_df_dict[name] = pd.DataFrame(columns=data_df_cols)

        self.intermediate_plot(0)

    def train_map_parameters(self):
        self.timer_obj = utils.ResumableTimer()
        self.timer_obj.start()

        for i in range(1, self.params.basis_empirical_iter + 1):
            self.glm.train_hyperparameters()
            self.glm.train_map()
            self.intermediate_plot(i)

    def intermediate_plot(self, i):
        self.glm.plot_covariates(self.data_df, self.evolution_df_dict, self.timer_obj)

    def callback(self, variable, status=None):
        if self.use_callbacks:
            print(variable)
