import numpy as np
import scipy.linalg as scla
import torch

from GLM.SparseGP import SparseGP, GpParams
from GLM.GLM_Model import Covariate
from collections import OrderedDict
from GLM.GLM_Model.BoundTransform import BoundTransform
from Utils import utils, TimeTracker, FilterParams
from opt_einsum import contract, contract_expression


class GP_Covariate(Covariate.Covariate):
    def __init__(self, params, etc_params, x, **property_dict):
        '''

        :param params:
        :param x: numpy array of covariate values
        :param property_dict:
        '''

        super().__init__(params, x, **property_dict)

        self.params_to_optimize = OrderedDict()
        self.main_params_to_optimize = OrderedDict()
        self.hyper_params_to_optimize = OrderedDict()
        self.all_params_to_optimize = OrderedDict()
        self.etc_params = etc_params
        self.gp_params = None
        self.gp_obj = None
        self.time = None

        self.bases_design_matrix = None
        self.bases_offset = None
        self.bases_mean = None
        self.bases_time = None
        self.bases_std = None

        self.exp_expression = None
        self.X_entire = None
        self.time_entire = None

    def add_gp_params(self, gp_params):
        self.gp_params = GpParams.GpParams(self.params, gp_params)

        for name, param in self.gp_params.gp_params.items():
            if param.requires_grad:
                self.params_to_optimize[name] = param

    def add_noise_param(self, glm_obj):
        self.gp_params.gp_params['kernel_epsilon_noise_std'].requires_grad = True
        glm_obj.register_parameter(name=f'{self.name}_kernel_epsilon_noise_std', param=self.gp_params.gp_params['kernel_epsilon_noise_std'])

    def update_gp_param_bounds(self):
        for param_name, param in self.gp_params.gp_params.items():
            self.gp_params.update_with_new_bounds(param_name, self.init_bounds_params[param_name])

    def add_time(self, time_params, glm_obj):
        self.time = TimeTracker.TimeTracker(self.params, **time_params)

        self.params_to_optimize['u'] = self.time.time_dict['u']
        self.main_params_to_optimize['u'] = self.time.time_dict['u']

        lower = self.init_bounds_params['u'][0]
        upper = self.init_bounds_params['u'][1]

        self.bounds_params['u'] = [lower, upper]
        self.time.initialize_transform(self.bounds_params['u'], 'u')
        glm_obj.register_parameter(name=f'{self.name}_u', param=self.time.time_dict['u'])

    def initialize_design_matrix(self):
        upper_add = 300 if self.name != 'History' else 100
        lower_add = 300 if self.name != 'History' else 0

        self.gp_obj = SparseGP.SparseGP(self.params, self.time, self.gp_params)
        self.X = torch.zeros((self.x.shape[1] * self.x.shape[0], self.time.duration), dtype=self.params.torch_d_type)
        self.X_entire = torch.zeros((self.x.shape[1] * self.x.shape[0], self.time.duration + upper_add + lower_add), dtype=self.params.torch_d_type)

        for i in range(self.x.shape[0]):
            design_matrix = utils.create_convolution_matrix(self.x[i, :], self.time.offset, self.time.duration)
            design_matrix_plot = utils.create_convolution_matrix(self.x[i, :], self.time.offset - lower_add, self.time.duration + upper_add + lower_add)
            X = torch.tensor(design_matrix, dtype=self.params.torch_d_type)
            X_entire = torch.tensor(design_matrix_plot, dtype=self.params.torch_d_type)
            self.X[self.x.shape[1] * i: self.x.shape[1] * (i + 1), :] = X
            self.X_entire[self.x.shape[1] * i: self.x.shape[1] * (i + 1), :] = X_entire

        self.exp_expression = contract_expression('...i,...i->...', (self.X.shape[0], self.X.shape[1]), self.X, constants=[1])
        self.X_test = self.X[-1 * self.params.num_test_trials * self.x.shape[1]:, :]
        self.X_test_entire = self.X_entire[-1 * self.params.num_test_trials * self.x.shape[1]:, :]
        self.X = self.X[:-1 * self.params.num_test_trials * self.x.shape[1], :]
        self.X_entire = self.X_entire[:-1 * self.params.num_test_trials * self.x.shape[1], :]
        self.X_sq = self.X**2

        return design_matrix

    def update_design_matrix_basis_form(self):
        duration = self.bases_design_matrix.shape[0]
        self.X = torch.zeros((self.x.shape[1] * self.x.shape[0], duration), dtype=self.params.torch_d_type)

        for i in range(self.x.shape[0]):
            design_matrix = utils.create_convolution_matrix(self.x[i, :], self.bases_offset, duration)
            X = torch.tensor(design_matrix, dtype=self.params.torch_d_type)
            self.X[self.x.shape[1] * i: self.x.shape[1] * (i + 1), :] = X

        self.X_test = self.X[-1 * self.params.num_test_trials * self.x.shape[1]:, :]
        self.X = self.X[:-1 * self.params.num_test_trials * self.x.shape[1], :]


    def initialize_filter_params(self, basis_df, glm_obj):
        relevant_df_dx = basis_df.shape[0] - 1
        r_hat = None

        h_basis = basis_df.loc[relevant_df_dx, 'entire_mean']
        c_basis = basis_df.loc[relevant_df_dx, 'entire_cov']
        h_basis_time = basis_df.loc[relevant_df_dx, 'entire_time']

        h_gp_time = self.time.time_dict['x']
        h_basis_min_dx = np.where(torch.tensor(h_basis_time, dtype=self.params.torch_d_type) >= h_gp_time[0])[0][0]
        h_basis_max_dx = np.where(torch.tensor(h_basis_time, dtype=self.params.torch_d_type) <= h_gp_time[-1])[0][-1]
        h_gp = h_basis[h_basis_min_dx: h_basis_max_dx + 1]
        c_gp = c_basis[h_basis_min_dx: h_basis_max_dx + 1, h_basis_min_dx: h_basis_max_dx + 1]
        self.time.time_plot = torch.tensor(basis_df.loc[relevant_df_dx, 'plot_time'], dtype=self.params.torch_d_type)


        if self.etc_params['use_basis_form']:
            # ard_coeff = torch.tensor(basis_df.loc[relevant_df_dx, 'entire_ard_coeff'], dtype=self.params.torch_d_type)
            self.bases_design_matrix = torch.tensor(basis_df.loc[relevant_df_dx, 'entire_basis'], dtype=self.params.torch_d_type)
            self.time.time_dict['x'] = torch.tensor(h_basis_time, dtype=self.params.torch_d_type)
            self.bases_std = 0.5 * torch.tensor(np.power(basis_df.loc[relevant_df_dx, 'entire_2std'], 2))
            self.bases_mean = torch.tensor(h_basis, dtype=self.params.torch_d_type)
            self.bases_time = torch.tensor(h_basis_time, dtype=self.params.torch_d_type)
            self.bases_offset = 1
            self.update_design_matrix_basis_form()
            # self.X.data = self.X.data @ self.bases_design_matrix

            return

        elif not self.etc_params['use_exp_mean'] and not self.etc_params['use_basis_form']:
            h_interpolated = self.get_initial_raw_mean(h_gp)
            r_hat = self.get_interpolated_covariance(c_gp)

        elif self.etc_params['use_exp_mean'] and not self.etc_params['use_basis_form']:
            h_interpolated = self.get_initial_exp_mean(h_gp)
            r_hat = self.get_interpolated_covariance(c_gp)

        # r_hat = np.identity(self.time.num_inducing)[self.time.triu_dx]
        variational_params = FilterParams.FilterParams(self.params, {'m': h_interpolated, 'r': r_hat})
        self.filter_params = variational_params

        for name, param in self.filter_params.filter_params.items():
            if param.requires_grad:
                self.params_to_optimize[name] = param
                self.main_params_to_optimize[name] = param

        self.all_params_to_optimize = self.params_to_optimize
        self.update_filter_params_with_transform(glm_obj)

    def get_initial_raw_mean(self, h_init):
        off_diag = 1e-1 if self.name != 'History' else 1e-5
        self.gp_obj.update_kernels()
        pseudo = (self.gp_obj['Kuu'] @ torch.inverse(
            self.gp_obj['Kxu'].t() @ self.gp_obj['Kxu'] + off_diag * torch.eye(self.time.num_inducing, dtype=self.params.torch_d_type)) @
                  self.gp_obj['Kxu'].t()).data.detach().numpy()
        h_time_u_interpolated = pseudo @ h_init[:self.time.time_dict['x'].shape[0]]

        return h_time_u_interpolated


    def get_initial_exp_mean(self, h_init):
        self.gp_obj.update_kernels()
        h = h_init[:self.time.time_dict['x'].shape[0]]
        most_neg_dx = np.argmin(h)
        next_neg_dx = most_neg_dx + 2 if (
                    h[most_neg_dx + 2] > h[most_neg_dx + 1] and h[most_neg_dx + 2] < 0) else most_neg_dx + 1
        tau = -(next_neg_dx - most_neg_dx) * self.params.delta / (np.log(h[next_neg_dx] / (h[most_neg_dx] - 2)))
        gain = (h[most_neg_dx] - 2)
        self.gp_params.update_with_transform_override_bounds(gain, 'gain')
        self.gp_params.update_with_transform_override_bounds(tau, 'tau')

        m_X = gain * torch.exp(-1 * self.time.time_dict['x'] / tau).data.detach().numpy()
        m_U = gain * torch.exp(-1 * self.time.time_dict_t['u']() / tau).data.detach().numpy()

        pseudo = (self.gp_obj['Kuu'] @ torch.inverse(self.gp_obj['Kxu'].t() @ self.gp_obj['Kxu'] + 1e-5 * torch.eye(self.time.num_inducing)) @ self.gp_obj['Kxu'].t()).data.detach().numpy()
        h_time_u_interpolated = pseudo @ (h - m_X) + m_U

        return h_time_u_interpolated

    def get_interpolated_covariance(self, c_gp):
        off_diag = 1e-1 if self.name != 'History' else 1e-3
        S_hat = c_gp

        inner = S_hat + (self.gp_obj['Kxu'] @ torch.inverse(self.gp_obj['Kuu']) @ self.gp_obj['Kxu'].t() - self.gp_obj['Kxx']).data.detach().numpy()
        outer = (self.gp_obj['Kuu'] @ torch.inverse(self.gp_obj['Kxu'].t() @ self.gp_obj['Kxu'] + off_diag * torch.eye(self.time.num_inducing)) @ self.gp_obj[
                     'Kxu'].t()).data.detach().numpy()

        Sigma_hat = outer @ inner @ outer.T + 1 * np.identity(self.time.num_inducing)
        r_hat = scla.cholesky(Sigma_hat, lower=False)
        r_hat = r_hat[self.time.triu_dx]

        return r_hat

    def initialize_gp_object(self):
        self.gp_obj = SparseGP.SparseGP(self.params, self.time, self.gp_params)

    def update_gp_params_with_transform(self, glm_obj):
        for name, param in self.gp_params.gp_params.items():
            if param.requires_grad:
                glm_obj.register_parameter(name=f'{self.name}_{name}_hyper', param=self.gp_params.gp_params[name])

            lower = self.init_bounds_params[name][0]
            upper = self.init_bounds_params[name][1]

            self.bounds_params[name] = [lower, upper]
            self.bounds_transform[name] = BoundTransform(self.params, [lower, upper])
            self.gp_params.update_with_transform(self.bounds_transform[name], name)

    def update_design_matrix(self):
        diff_upper = 2 * int(np.ceil((self.time.time_dict['torch']['u'].data.detach().numpy().max() -
                                      self.time.time_dict['torch']['x'].data.numpy().max()) / self.delta))
        diff_lower = 2 * int(np.ceil((self.time.time_dict['torch']['x'].data.numpy().min() -
                                      self.time.time_dict['torch']['u'].data.detach().numpy().min()) / self.delta))

        if self.name is 'History':
            return

        num_add_upper = diff_upper if diff_upper > 0 else 0
        num_add_lower = diff_lower if diff_lower > 0 else 0

        if num_add_upper > num_add_lower:
            num_add_lower = num_add_upper

        elif num_add_upper < num_add_lower:
            num_add_upper = num_add_lower

        if num_add_upper >= 50:
            num_add_lower = 50
            num_add_upper = 50
        if num_add_upper == 0 and num_add_lower == 0:
            return

        new_offset = self.time.offset - num_add_lower
        new_duration = self.time.duration + num_add_lower + num_add_upper

        upper_times = self.time.time_dict['torch']['x'].data.numpy().max() + self.delta * np.arange(1, num_add_upper + 1)
        lower_times = self.time.time_dict['torch']['x'].data.numpy().min() - self.delta * np.arange(1, num_add_lower + 1)

        if self.name is 'History':
            lower_times = np.array([])
        new_times = np.concatenate([lower_times, upper_times])

        self.X = torch.tensor(utils.create_convolution_matrix(self.x, new_offset, new_duration), dtype=torch.double)
        self.time.add_design_matrix_points(new_times)
        self.time.offset = new_offset
        self.time.duration = new_duration

    def get_log_likelihood_terms(self):
        if self.etc_params['use_basis_form']:
            likelihood_mu_term = self.X @ self.bases_mean
            exp_arg = self.X @ self.bases_mean
            kld_term = torch.zeros(1, dtype=self.params.torch_d_type)

        else:
            R = torch.zeros((self.time.time_dict['u'].shape[0], self.time.time_dict['u'].shape[0]), dtype=self.params.torch_d_type)
            R[self.time.triu_dx] = self.filter_params.filter_params_t['r']()

            if not self.etc_params['use_exp_mean']:
                mu_q_x = self.gp_obj['Kxu'] @ self.gp_obj['Kuu_inv'] @ self.filter_params.filter_params_t['m']()
            else:
                mu_q_x = (self.gp_params.gp_params_t['gain']() * torch.exp(-1*self.time.time_dict['x']/self.gp_params.gp_params_t['tau']()) +
                          self.gp_obj['Kxu'] @ self.gp_obj['Kuu_inv'] @
                          (self.filter_params.filter_params_t['m']() - self.gp_params.gp_params_t['gain']() * torch.exp(-1*self.time.time_dict_t['u']()/self.gp_params.gp_params_t['tau']())))

            sigma_q_x = (self.gp_obj['Kxx'] - self.gp_obj['Kxu'] @ self.gp_obj['Kuu_inv'] @ self.gp_obj['Kxu'].t()
                         + self.gp_obj['Kxu'] @ self.gp_obj['Kuu_inv'] @ R.t() @ R @ self.gp_obj['Kuu_inv'] @
                         self.gp_obj['Kxu'].t())

            likelihood_mu_term = self.X @ mu_q_x

            # exp_diag = torch.zeros(self.X.shape[0])
            # for i in range(self.X.shape[0]):
            #     exp_diag[i].data = self.X[i,:] @ sigma_q_x @ self.X[i,:]
            # torch.einsum('ij, jk, ik -> i', self.X, sigma_q_x, self.X)
            # exp_arg = (self.X @ mu_q_x + 0.5 * torch.diag(self.X @ sigma_q_x @ self.X.t()))

            # exp_arg = self.X @ mu_q_x + 0.5 * torch.einsum('...i,...i->...', self.X @ sigma_q_x, self.X)
            # exp_arg = self.X @ mu_q_x + 0.5 * self.exp_expression(self.X @ sigma_q_x)
            # zzz = torch.sum(self.X_sq * torch.diag(sigma_q_x), axis=1)
            exp_arg = self.X @ mu_q_x + 0.5 * torch.sum(self.X_sq * torch.diag(sigma_q_x), axis=1)



            if not self.etc_params['use_exp_mean']:
                kld_term = -0.5 * (torch.trace(self.gp_obj['Kuu_inv'] @ R.t() @ R) + torch.logdet(self.gp_obj['Kuu']) - torch.logdet(R.t() @ R)
                               + self.filter_params.filter_params_t['m']() @ (self.gp_obj['Kuu_inv'] @ self.filter_params.filter_params_t['m']()))
            else:
                kld_term = -0.5 * (torch.trace(self.gp_obj['Kuu_inv'] @ R.t() @ R) + torch.logdet(self.gp_obj['Kuu']) - torch.logdet(R.t() @ R)
                                   + (self.filter_params.filter_params_t['m']() - self.gp_params.gp_params_t['gain']() * torch.exp(-1 * self.time.time_dict_t['u']() / self.gp_params.gp_params_t['tau']()))
                                      @ (self.gp_obj['Kuu_inv'] @ (self.filter_params.filter_params_t['m']() - self.gp_params.gp_params_t['gain']() * torch.exp(-1 * self.time.time_dict_t['u']() / self.gp_params.gp_params_t['tau']()))))

        return likelihood_mu_term, exp_arg, kld_term

    def loss(self):
        if self.etc_params['use_basis_form']:
            likelihood_mu_term = self.X @ self.bases_mean
            exp_arg = self.X @ self.bases_mean

        else:
            entire_mean, entire_covariance = self.get_entire_values()
            likelihood_mu_term = self.X_entire @ entire_mean
            exp_arg = self.X_entire @ entire_mean

        return likelihood_mu_term, exp_arg

    def test_loss(self):
        if self.etc_params['use_basis_form']:
            likelihood_mu_term = self.X_test @ self.bases_mean
            exp_arg = self.X_test @ self.bases_mean

        else:
            entire_mean, entire_covariance = self.get_entire_values()
            likelihood_mu_term = self.X_test_entire @ entire_mean
            exp_arg = self.X_test_entire @ entire_mean

        return likelihood_mu_term, exp_arg

    def get_entire_values(self):
        self.set_entire_times()
        R = torch.zeros((self.time.time_dict['u'].shape[0], self.time.time_dict['u'].shape[0]), dtype=self.params.torch_d_type)
        R[self.time.triu_dx] = self.filter_params.filter_params_t['r']()

        Kxx = self.gp_params.gp_kernel['kernel_fn'](self.time_entire, self.time_entire, self.params,
                                                    **self.gp_params.gp_params_t)
        Kxu = self.gp_params.gp_kernel['kernel_fn'](self.time_entire, self.time.time_dict_t['u'](), self.params,
                                                    **self.gp_params.gp_params_t)

        entire_covariance = (Kxx - Kxu @ self.gp_obj['Kuu_inv'] @ Kxu.T +
                             Kxu @ self.gp_obj['Kuu_inv'] @ R.t() @ R @ self.gp_obj['Kuu_inv'] @ Kxu.T).detach().numpy()

        if not self.etc_params['use_exp_mean']:
            entire_mean = Kxu @ self.gp_obj['Kuu_inv'] @ self.filter_params.filter_params_t['m']()
        else:
            entire_mean = (self.gp_params.gp_params_t['gain']() * torch.exp(
                -1 * self.time_entire / self.gp_params.gp_params_t['tau']()) + Kxu @ self.gp_obj['Kuu_inv'] @ (self.filter_params.filter_params_t['m']() -
                            self.gp_params.gp_params_t['gain']() * torch.exp(-1 * self.time.time_dict_t['u']() / self.gp_params.gp_params_t['tau']())))


        return entire_mean, entire_covariance


    def get_values_to_plot(self):
        if self.etc_params['use_basis_form']:
            plot_mean = self.bases_mean.data.detach().numpy()
            plot_std = self.bases_std.data.detach().numpy()
            plot_time = self.bases_time.data.detach().numpy()

            entire_mean = self.bases_mean.data.detach().numpy()
            entire_std = self.bases_std.data.detach().numpy()
            entire_time = self.bases_time.data.detach().numpy()

        else:
            with torch.no_grad():
                R = torch.zeros((self.time.time_dict['u'].shape[0], self.time.time_dict['u'].shape[0]), dtype=self.params.torch_d_type)
                R[self.time.triu_dx] = self.filter_params.filter_params_t['r']()

                Kp = self.gp_params.gp_kernel['kernel_fn'](self.time.time_plot, self.time.time_plot, self.params, **self.gp_params.gp_params_t)
                Kpu = self.gp_params.gp_kernel['kernel_fn'](self.time.time_plot, self.time.time_dict_t['u'](), self.params, **self.gp_params.gp_params_t)

                covariance_plot = (Kp - Kpu @ self.gp_obj['Kuu_inv'] @ Kpu.T +
                                   Kpu @ self.gp_obj['Kuu_inv'] @ R.t() @ R @ self.gp_obj['Kuu_inv'] @ Kpu.T).detach().numpy()

                if not self.etc_params['use_exp_mean']:
                    mean_plot = Kpu @ self.gp_obj['Kuu_inv'] @ self.filter_params.filter_params_t['m']()
                else:
                    mean_plot = (self.gp_params.gp_params_t['gain']() * torch.exp(-1*self.time.time_plot/self.gp_params.gp_params_t['tau']()) +
                              Kpu @ self.gp_obj['Kuu_inv'] @
                              (self.filter_params.filter_params_t['m']() - self.gp_params.gp_params_t['gain']() * torch.exp(-1*self.time.time_dict_t['u']()/self.gp_params.gp_params_t['tau']())))

                plot_mean = mean_plot.data.detach().numpy()
                plot_std = np.sqrt(np.diag(covariance_plot))
                plot_time = self.time.time_plot.data.detach().numpy()

                entire_mean, entire_covariance = self.get_entire_values()

                entire_mean = entire_mean.data.detach().numpy()
                entire_std = np.sqrt(np.diag(entire_covariance))
                entire_time = self.time_entire.data.detach().numpy()

        return plot_mean, plot_std, plot_time, entire_mean, entire_std, entire_time

    def set_entire_times(self):
        with torch.no_grad():
            if self.name == 'History':
                time_entire = torch.cat([self.time.time_dict['x'].clone(),
                                         torch.arange(torch.round(10**3 * self.time.time_dict['x'].max()) / 10**3 + self.params.delta,
                                                      torch.round(10**3 * self.time.time_dict['x'].max()) / 10**3 + 100 * self.params.delta,
                                                      self.params.delta, dtype=self.params.torch_d_type)])
            else:
                add_factor = 300

                time_entire = torch.cat([self.time.time_dict['x'].data,
                                         torch.arange(self.time.time_dict['x'].max().data + self.params.delta,
                                                      self.time.time_dict['x'].max().data + add_factor * self.params.delta + 1e-12,
                                                      self.params.delta, dtype=self.params.torch_d_type)])

                lower_append = torch.arange(self.time.time_dict['x'].min().data - add_factor * self.params.delta,
                                                      self.time.time_dict['x'].min().data,
                                                      self.params.delta, dtype=self.params.torch_d_type)



                if lower_append[-1] == self.time.time_dict['x'].min().data:
                    lower_append = lower_append[:-1]
                elif np.abs(lower_append[-1] - self.time.time_dict['x'].min()) < 1e-6:
                    lower_append = lower_append[:-1]

                time_entire = torch.cat([lower_append, time_entire])

            self.time_entire = time_entire
            # old_time_plot_min = np.array([self.time.time_plot.min()])
            # old_time_plot_max = np.array([self.time.time_plot.max()])
            #
            # time_u_min = self.time.time_dict_t['u']().data.detach().numpy().min()
            # time_u_max = self.time.time_dict_t['u']().data.detach().numpy().max()
            #
            # time_plot_min = min(old_time_plot_min, time_u_min)
            # time_plot_max = max(old_time_plot_max, time_u_max)
            #
            # self.time.time_plot_min = time_plot_min
            # self.time.time_plot_max = time_plot_max
            # self.time.time_plot = torch.tensor(np.arange(time_plot_min, time_plot_max + 1e-12, self.delta), dtype=self.params.torch_d_type)






