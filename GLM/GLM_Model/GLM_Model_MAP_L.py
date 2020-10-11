import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import scipy

from GLM.GLM_Model import GLM_Model, PyTorchObj
from scipy.optimize import minimize, Bounds
from tqdm import tqdm


class GLM_Model_MAP(GLM_Model.GLM_Model):
    def __init__(self, params):
        super().__init__(params)
        self.training_parameters = None

    def train_map(self):
        params_to_optimize = [param for param in self.state_dict().keys() if '_m' in param]
        params_to_optimize.append('baseline')
        self.set_optimizing_parameters_grad(params_to_optimize)
        self.initialize_covariate_means()

        self.set_training_parameters(params_to_optimize)
        # self.optimizer = torch.optim.LBFGS(self.training_parameters, lr=1, history_size=20, max_iter=self.params.map_iter, line_search_fn='strong_wolfe')
        # optimizer_closure = self.map_closure()
        #
        # print(self.optimizer.step(optimizer_closure))

        maxiter = self.params.basis_map_iter

        with tqdm(total=maxiter) as pbar:
            def verbose(xk):
                pbar.update(1)

            obj = PyTorchObj.PyTorchObjective(self, params_to_optimize, self.scipy_map_closure)
            xL = scipy.optimize.minimize(obj.fun, obj.x0, method='TNC', jac=obj.jac, callback=verbose,
                                         options={'gtol': 1e-6, 'disp': True,
                                                  'maxiter': maxiter})

        print('done')

    def initialize_covariate_means(self):
        for covariate_name, covariate in self.covariates.items():
            where = torch.where(torch.sqrt(covariate.filter_params.filter_params_t['r']()) < 5e-2)

            if where[0].shape[0] > 0:
                print(f'{where[0].shape[0]} eliminated in {covariate_name}')
            covariate.filter_params.filter_params['m'].data[where[0]] = torch.tensor(np.array(0), dtype=self.params.torch_d_type)

    def scipy_hyper_closure(self):
        self.zero_grad()
        loss = self.n_marginal_likelihood()

        return loss

    def scipy_map_closure(self):
        self.zero_grad()
        loss = self.get_nlog_likelihood()

        return loss

    def map_closure(self):
        def closure():
            self.optimizer.zero_grad()
            loss = self.get_nlog_likelihood()
            loss.backward()

            return loss

        return closure

    def train_hyperparameters(self):
        params_to_optimize = [param for param in self.state_dict().keys() if '_r' in param]
        self.set_optimizing_parameters_grad(params_to_optimize)

        self.set_training_parameters(params_to_optimize)
        # self.optimizer = torch.optim.LBFGS(self.training_parameters, lr=0.1, history_size=20, max_iter=self.params.basis_evidence_iter, line_search_fn='strong_wolfe')
        # optimizer_closure = self.hyperparameter_closure()

        # print(self.optimizer.step(optimizer_closure))

        maxiter = self.params.basis_evidence_iter

        with tqdm(total=maxiter) as pbar:
            def verbose(xk):
                pbar.update(1)

            obj = PyTorchObj.PyTorchObjective(self, params_to_optimize, self.scipy_hyper_closure)
            xL = scipy.optimize.minimize(obj.fun, obj.x0, method='TNC', jac=obj.jac, callback=verbose,
                                         options={'gtol': 1e-6, 'disp': True,
                                                  'maxiter': maxiter})

        print('done')

    def hyperparameter_closure(self):
        def closure():
            self.optimizer.zero_grad()
            loss = self.n_marginal_likelihood()
            loss.backward()

            return loss

        return closure

    def get_nlog_likelihood(self, for_saving=False):
        total_likelihood = torch.zeros(1, dtype=self.params.torch_d_type)
        total_exp = torch.zeros(self.y.shape[0], dtype=self.params.torch_d_type)
        total_kld = torch.zeros(1, dtype=self.params.torch_d_type)

        for covariate_name, cov in self.covariates.items():
            ll, e_arg, gaussian_term = cov.get_log_likelihood_terms()

            total_likelihood += self.y @ ll
            total_exp += e_arg
            total_kld += gaussian_term

        total_exp = torch.sum(torch.exp(total_exp + self.baseline * torch.ones(self.y.shape[0], dtype=self.params.torch_d_type)))
        total_likelihood = total_likelihood + self.y @ (self.baseline * torch.ones(self.y.shape[0], dtype=self.params.torch_d_type))
        nll = -1 * (total_likelihood - self.params.delta * total_exp)

        if not for_saving:
            nll = nll - total_kld

        return nll

    def n_marginal_likelihood(self):
        nll = self.get_nlog_likelihood()
        logdet_term_pos = self.get_laplace_logdet_term()
        loss = nll - logdet_term_pos

        return loss

    def get_laplace_logdet_term(self):
        logdet_term = -0.5 * torch.logdet(self.get_inverse_posterior_covariance())

        return logdet_term

    def get_inverse_posterior_covariance(self):
        for dx, (name, cov) in enumerate(self.covariates.items()):
            if dx == 0:
                big_X = cov.X
                big_h = cov.filter_params.filter_params_t['m']()
                big_r = cov.filter_params.filter_params_t['r']()
            else:
                big_X = torch.cat([big_X, cov.X], axis=1)
                big_h = torch.cat([big_h, cov.filter_params.filter_params_t['m']()])
                big_r = torch.cat([big_r, cov.filter_params.filter_params_t['r']()])

        big_baseline = self.baseline * torch.ones(self.y.shape[0], dtype=self.params.torch_d_type)
        inverse_posterior_covariance = self.delta * (big_X.t() * torch.exp(big_X @ big_h + big_baseline)) @ big_X + torch.diag(1 / big_r)

        return inverse_posterior_covariance

    def plot_covariate(self, name, axs):
        exp_term = self.get_exp_term_for_posterior_covariance()
        m_plot, r_plot, noise_plot = self.covariates[name].get_values_to_plot(exp_term)

        new_time, new_m, new_noise = zip(*sorted(zip(self.delta * np.arange(1, m_plot.shape[0]), m_plot, noise_plot)))
        new_time = np.array(new_time)
        new_m = np.array(new_m)
        new_noise = np.array(new_noise)

        if name == 'History':
            end_dx = 60
        else:
            end_dx = m_plot.shape[0]
        axs.plot(self.delta * np.arange(1, m_plot.shape[0])[0:end_dx], new_m[0:end_dx], label='posterior mean', color='royalblue')
        axs.fill_between(new_time[0:end_dx], new_m[0:end_dx] - 2 * new_noise[0:end_dx],
                         new_m[0:end_dx] + 2 * new_noise[0:end_dx], alpha=0.3, color='cornflowerblue')

        return new_time, new_m, r_plot, new_noise

    def get_exp_term_for_posterior_covariance(self):

        with torch.no_grad():
            exp_arg = torch.zeros(self.y.shape[0], dtype=self.params.torch_d_type)

            for name, covariate in self.covariates.items():
                exp_arg += covariate.X @ covariate.filter_params.filter_params_t['m']()

            exp_arg += self.baseline + torch.ones(self.y.shape[0], dtype=self.params.torch_d_type)

        return exp_arg

    def plot_covariates(self, data_df, evolution_df_dict):
        with torch.no_grad():
            nll = self.get_nlog_likelihood(for_saving=True)
            plt.style.use("ggplot")
            fig, axs = plt.subplots(2, int(np.ceil(len(self.covariates.keys())/2)), figsize=(10, 5))
            axs = axs.flatten()

            parameter_beg_dx = 0
            inverse_posterior_variance = torch.inverse(self.get_inverse_posterior_covariance())

            for dx, (name, covariate) in enumerate(self.covariates.items()):
                parameter_end_dx = parameter_beg_dx + covariate.filter_params.filter_params['m'].shape[0]
                covariate_covariance = inverse_posterior_variance[parameter_beg_dx: parameter_end_dx, parameter_beg_dx: parameter_end_dx]
                entire_mean, entire_2std, entire_time, plot_mean, plot_2std, plot_time = covariate.get_values_to_plot(covariate_covariance)
                parameter_beg_dx += covariate.filter_params.filter_params['m'].shape[0]

                axs[dx].plot(plot_time, plot_mean, label='posterior mean', color='royalblue')
                axs[dx].fill_between(plot_time, plot_mean - plot_2std, plot_mean + plot_2std, alpha=0.3, color='cornflowerblue')
                axs[dx].set_ylim([plot_mean.min() - 1, plot_mean.max() + 1])
                axs[dx].set_title(name)

                ev_dx = evolution_df_dict[name].shape[0]
                evolution_df_dict[name].at[ev_dx, 'plot_mean'] = np.copy(plot_mean)
                evolution_df_dict[name].at[ev_dx, 'plot_2std'] = np.copy(plot_2std)
                evolution_df_dict[name].at[ev_dx, 'plot_time'] = np.copy(plot_time)
                evolution_df_dict[name].at[ev_dx, 'entire_mean'] = np.copy(entire_mean)
                evolution_df_dict[name].at[ev_dx, 'entire_2std'] = np.copy(entire_2std)
                evolution_df_dict[name].at[ev_dx, 'entire_time'] = np.copy(entire_time)
                evolution_df_dict[name].at[ev_dx, 'entire_basis'] = np.copy(covariate.B.data.detach().numpy())
                evolution_df_dict[name].at[ev_dx, 'entire_coeff'] = np.copy(covariate.filter_params.filter_params_t['m']().data.detach().numpy())
                evolution_df_dict[name].at[ev_dx, 'entire_ard_coeff'] = np.copy(covariate.filter_params.filter_params_t['r']().data.detach().numpy())
                evolution_df_dict[name].at[ev_dx, 'nll'] = np.copy(nll)
                evolution_df_dict[name].at[ev_dx, 'baseline'] = np.copy(self.baseline.data.detach().numpy())
                evolution_df_dict[name].to_pickle(f'{self.params.basis_ev_path}_{name}')

        plt.legend()
        plt.subplots_adjust(hspace=2.0)
        print(f'nll: {nll}')
        plt.savefig(self.params.basis_filter_plot_path, dpi=300)
        plt.show()






