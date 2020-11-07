import pandas as pd
from collections import OrderedDict
from Utils import utils


class PoissonVariational:
    def __init__(self, params, data_df, glm, kernel_prep_dict, use_callbacks=False):
        self.params = params
        self.delta = params.delta
        self.glm = glm
        self.data_df = data_df
        self.kernel_prep_dict = kernel_prep_dict
        self.time_plot = None
        self.optimize_hyper = False

        self.evolution_df_dict = OrderedDict()
        self.optimizer_params = None
        self.use_callbacks = use_callbacks
        self.optimize_over = OrderedDict()
        self.optimize_these_params = OrderedDict()

    def initialize_variational_model(self):
        self.glm.initialize_covariates()

        for name, cov_s in self.data_df.iterrows():
            data_df_cols = ['plot_time', 'plot_2std', 'plot_mean', 'entire_time', 'entire_2std', 'entire_mean', 'baseline',
                            'nll', 'nll_test', 'time_so_far',
                            'mean_color', 'std_color']

            self.evolution_df_dict[name] = pd.DataFrame(columns=data_df_cols)

    def train_variational_parameters(self):
        self.timer_obj = utils.ResumableTimer()
        self.timer_obj.start()
        for i in range(1, self.params.basis_empirical_iter + 1):
            self.glm.train_hyperparameters(self.kernel_prep_dict, i)
            self.glm.train_variational_parameters(self.kernel_prep_dict, i)
            self.intermediate_plot()

    def intermediate_plot(self):
        self.glm.update_covariate_gp_objects()
        self.glm.plot_covariates(self.evolution_df_dict, self.timer_obj)

