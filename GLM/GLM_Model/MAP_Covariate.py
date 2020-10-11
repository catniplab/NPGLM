import numpy as np
import matplotlib.pyplot as plt
import torch

from GLM.GLM_Model import Covariate
from Utils import utils, TimeTracker, FilterParams


class MAP_Covariate(Covariate.Covariate):
    def __init__(self, params, x, **property_dict):
        '''

        :param params:
        :param x: numpy array of covariate values
        :param property_dict:
        '''
        super().__init__(params, x, **property_dict)

        self.B = None
        self.bases_params = None

    def add_bases_params(self, bases_params):
        self.bases_params = bases_params

    def initialize_filter_params(self, h_ml):

        filter_params = FilterParams.FilterParams(self.params, {'m': h_ml, 'r': np.ones(h_ml.shape[0])})
        self.filter_params = filter_params

        for name, param in self.filter_params.filter_params.items():
            if param.requires_grad:
                self.params_to_optimize[name] = param

    def get_log_likelihood_terms(self):
        likelihood_term = self.X @ self.filter_params.filter_params_t['m']()

        exp_arg = self.X @ self.filter_params.filter_params_t['m']()

        kld_term = -0.5 * (torch.sum(torch.log(self.filter_params.filter_params_t['r']())) +
                           self.filter_params.filter_params_t['m']() @
                           torch.diag(1 / self.filter_params.filter_params_t['r']()) @
                           self.filter_params.filter_params_t['m']())

        return likelihood_term, exp_arg, kld_term

    def get_test_log_likelihood_terms(self):
        likelihood_term = self.X_test @ self.filter_params.filter_params_t['m']()

        exp_arg = self.X_test @ self.filter_params.filter_params_t['m']()

        kld_term = -0.5 * (torch.sum(torch.log(self.filter_params.filter_params_t['r']())) +
                           self.filter_params.filter_params_t['m']() @
                           torch.diag(1 / self.filter_params.filter_params_t['r']()) @
                           self.filter_params.filter_params_t['m']())

        return likelihood_term, exp_arg, kld_term

    def initialize_design_matrix(self):
        '''
        For MAP inference the design matrix is
        :return:
        '''

        bases_matrix = self.bases_params['bases_fn'](**self.bases_params).T
        self.B = torch.tensor(bases_matrix, dtype=self.params.torch_d_type)
        self.X = torch.zeros((self.x.shape[1] * self.x.shape[0], self.B.shape[1]), dtype=self.params.torch_d_type)

        for i in range(self.x.shape[0]):
            design_matrix = utils.create_convolution_matrix(self.x[i, :], self.bases_params['offset'], self.B.shape[0])
            X = torch.tensor(design_matrix, dtype=self.params.torch_d_type)
            self.X[self.x.shape[1] * i: self.x.shape[1] * (i+1), :] = X @ self.B

        for i in range(self.B.shape[1]):
            plt.plot(self.B[:,i])

        plt.title(f'{self.name} Bases')
        plt.savefig(self.params.basis_filter_plot_path + f'_{self.name}_basis_fn.pdf')
        plt.show()

        self.X_test = self.X[-1 * self.params.num_test_trials * self.x.shape[1]:, :]
        self.X = self.X[: -1 * self.params.num_test_trials * self.x.shape[1], :]


        return self.X

    def get_values_to_plot(self, this_cov_variance):
        with torch.no_grad():
            beg_dx = self.bases_params['time_plot_min'] - self.bases_params['offset']
            end_dx = beg_dx + (self.bases_params['time_plot_max'] - self.bases_params['time_plot_min'])

            entire_mean = self.B @ self.filter_params.filter_params_t['m']()
            entire_cov = self.B @ this_cov_variance @ self.B.t()
            entire_2std = 2 * torch.sqrt(torch.diag(entire_cov))
            entire_time = self.params.delta * np.arange(self.bases_params['offset'], self.bases_params['offset'] + self.B.shape[0])

            entire_mean = entire_mean.data.detach().numpy()
            entire_2std = entire_2std.data.detach().numpy()

            plot_mean = entire_mean[beg_dx: end_dx]
            plot_2std = entire_2std[beg_dx: end_dx]
            plot_time = entire_time[beg_dx: end_dx]

        return entire_mean, entire_2std, entire_time, plot_mean, plot_2std, plot_time, entire_cov

    def get_map_covariance(self, exp_arg):
        coefficient_cov = torch.inverse(
            torch.diag(1 / self.filter_params.filter_params_torch['r']) + self.delta * self.X.t() @ (torch.diag(torch.exp(exp_arg)) @ self.X))
        filter_cov = self.B @ coefficient_cov @ self.B.t()
        filter_cov = filter_cov.data.detach().numpy()

        return filter_cov


