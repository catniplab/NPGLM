import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy

from GLM.GLM_Model import GLM_Model, PyTorchObj
from scipy.optimize import minimize, Bounds
from tqdm import tqdm


class GLM_Model_GP(GLM_Model.GLM_Model):
    def __init__(self, params):
        super().__init__(params)

        self.kernel_prep_dict = None
        self.first_time_train_this_covariate = None
        self.covariate_training = None
        self.total_likelihood = None
        self.total_exp = None
        self.total_kld = None

    def add_covariate(self, covariate):
        super().add_covariate(covariate)
        self.register_parameter(name=f'{covariate.name}_u', param=covariate.time.time_dict['u'])

    def train_variational_parameters(self, kernel_prep_dict, i):
        self.kernel_prep_dict = kernel_prep_dict
        self.update_time_bounds()

        for covariate_name, covariate in self.covariates.items():
            if covariate.etc_params['use_basis_form']:
                continue
            if i <= 2 or (i > 2 and i % 2 == 0):
                params_to_optimize = [param for param in self.state_dict().keys() if (param.startswith(covariate_name) and
                                                                                  not (param.endswith('_hyper'))) and not (param.endswith('_u'))]
            else:
                params_to_optimize = [param for param in self.state_dict().keys() if (param.startswith(covariate_name) and
                                                                                  not (param.endswith('_hyper')))]

            params_to_optimize.append('baseline')

            for name, param in self.named_parameters():
                if name not in params_to_optimize:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            self.update_covariate_gp_objects()
            self.set_training_parameters(params_to_optimize)
            # self.optimizer = torch.optim.LBFGS(self.training_parameters, lr=1, history_size=10, max_iter=self.params.gp_variational_iter, line_search_fn='strong_wolfe')
            # optimizer_closure = self.nll_closure()
            self.first_time_train_this_covariate = True
            self.covariate_training = covariate_name
            self.total_likelihood = torch.zeros(1, dtype=self.params.torch_d_type)
            self.total_exp = torch.zeros(self.y.shape[0], dtype=self.params.torch_d_type)
            self.total_kld = torch.zeros(1, dtype=self.params.torch_d_type)

            maxiter = self.params.gp_variational_iter
            with tqdm(total=maxiter) as pbar:
                def verbose(xk):
                    pbar.update(1)

                obj = PyTorchObj.PyTorchObjective(self, params_to_optimize, self.scipy_closure)
                xL = scipy.optimize.minimize(obj.fun, obj.x0, method='TNC', jac=obj.jac, callback=verbose,
                                             options={'gtol': 1e-6, 'disp': True,
                                                      'maxiter': maxiter})

        print('done')

    def add_noise_parameter(self):
        for covariate_name, covariate in self.covariates.items():
            if covariate.etc_params['use_basis_form']:
                continue

            covariate.add_noise_param(self)

    def train_hyperparameters(self, kernel_prep_dict, i):
        self.kernel_prep_dict = kernel_prep_dict
        self.update_gp_param_bounds()


        if i > 4:
            self.add_noise_parameter()

        for covariate_name, covariate in self.covariates.items():
            if covariate.etc_params['use_basis_form']:
                continue

            params_to_optimize = [param for param in self.state_dict().keys() if (param.startswith(covariate_name) and
                                                                                  param.endswith('_hyper'))]

            for name, param in self.named_parameters():
                if name not in params_to_optimize:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        # params_to_optimize = [param for param in self.state_dict().keys() if (not param.startswith('History') and
        #                                                                       param.endswith('_hyper'))]
            self.update_covariate_gp_objects()
            self.set_training_parameters(params_to_optimize)
            # self.optimizer = torch.optim.LBFGS(self.training_parameters, lr=0.3, history_size=5, max_iter=self.params.gp_hyperparameter_iter, line_search_fn='strong_wolfe')
            self.first_time_train_this_covariate = True
            self.covariate_training = covariate_name
            self.total_likelihood = torch.zeros(1, dtype=self.params.torch_d_type)
            self.total_exp = torch.zeros(self.y.shape[0], dtype=self.params.torch_d_type)
            self.total_kld = torch.zeros(1, dtype=self.params.torch_d_type)
            # optimizer_closure = self.nll_closure_hyper()
            # self.zero_grad()
            # print(self.optimizer.step(optimizer_closure))
            maxiter = self.params.gp_hyperparameter_iter
            with tqdm(total=maxiter) as pbar:
                def verbose(xk):
                    pbar.update(1)

                obj = PyTorchObj.PyTorchObjective(self, params_to_optimize, self.scipy_closure)
                xL = scipy.optimize.minimize(obj.fun, obj.x0, method='TNC', jac=obj.jac, callback=verbose,
                                       options={'gtol': 1e-6, 'disp': True,
                                                'maxiter': maxiter})

        print('done')

    def scipy_closure(self):
        self.zero_grad()
        # TODO
        self.update_covariate_gp_objects(update_all=False)
        loss = self.get_nlog_likelihood()

        return loss

    def nll_closure(self):
        def closure():
            self.optimizer.zero_grad()
            # TODO
            self.update_covariate_gp_objects(update_all=False)
            loss = self.get_nlog_likelihood()
            loss.backward()

            return loss

        return closure

    def nll_closure_hyper(self):
        def closure():
            self.optimizer.zero_grad()
            # TODO
            self.update_covariate_gp_objects(update_all=False)
            loss = self.get_nlog_likelihood()
            loss.backward()

            return loss

        return closure

    def update_covariate_gp_objects(self, update_all=True):
        if update_all:
            with torch.no_grad():
                for covariate_name, covariate in self.covariates.items():
                    covariate.gp_obj.update_kernels()
                    covariate.gp_obj.compute_needed_chol_and_inv(self.kernel_prep_dict)
                    self.zero_grad()
        else:

            self.covariates[self.covariate_training].gp_obj.update_kernels()
            self.covariates[self.covariate_training].gp_obj.compute_needed_chol_and_inv(self.kernel_prep_dict)

    def update_gp_param_bounds(self):
        for covariate_name, covariate in self.covariates.items():
            covariate.update_gp_param_bounds()

    def update_time_bounds(self):
        for covariate_name, covariate in self.covariates.items():
            covariate.time.update_with_new_bounds('u')

    def update_covariate_design_matrices(self):
        for covariate_name, covariate in self.covariates.items():
            covariate.update_design_matrix()

    def get_nlog_likelihood(self, optimize_hyper=False):
        total_likelihood = torch.zeros(1, dtype=self.params.torch_d_type)
        total_exp = torch.zeros(self.y.shape[0], dtype=self.params.torch_d_type)
        total_kld = torch.zeros(1, dtype=self.params.torch_d_type)

        for covariate_name, cov in self.covariates.items():
            if covariate_name != self.covariate_training and not self.first_time_train_this_covariate:
                continue

            ll, e_arg, gaussian_term = cov.get_log_likelihood_terms()

            total_likelihood += self.y @ ll
            total_exp += e_arg
            total_kld += gaussian_term

            if covariate_name != self.covariate_training and self.first_time_train_this_covariate:
                self.total_likelihood += self.y @ ll
                self.total_exp += e_arg
                self.total_kld += gaussian_term

        if self.first_time_train_this_covariate:
            total_exp = torch.sum(torch.exp(total_exp + self.baseline * torch.ones(self.y.shape[0], dtype=self.params.torch_d_type)))
            total_likelihood = total_likelihood + self.y @ (self.baseline * torch.ones(self.y.shape[0], dtype=self.params.torch_d_type))
            nll = -1 * (total_likelihood - self.params.delta * total_exp + total_kld)
            self.first_time_train_this_covariate = False
        else:
            total_exp = torch.sum(torch.exp(total_exp + self.total_exp + self.baseline * torch.ones(self.y.shape[0], dtype=self.params.torch_d_type)))
            total_likelihood = total_likelihood + self.total_likelihood + self.y @ (self.baseline * torch.ones(self.y.shape[0], dtype=self.params.torch_d_type))
            nll = -1 * (total_likelihood - self.params.delta * total_exp + total_kld + self.total_kld)

        return nll

    def get_nats_per_bin(self, y, exp_arg):
        lambda_0 = torch.sum(y) / (y.shape[0] * self.params.delta)

        nats_per_bin = y * exp_arg - self.params.delta * torch.exp(exp_arg)
        nats_per_bin = nats_per_bin - (y * torch.log(lambda_0) - self.params.delta * lambda_0 * torch.ones_like(y, dtype=self.params.torch_d_type))

        # nats_per_bin = nats_per_bin  - (y * np.log(lambda_0) - self.params.delta * lambda_0 * np.ones_like(y))
        total_num_spikes = torch.sum(y)
        nll_test_mean = torch.sum(nats_per_bin) / total_num_spikes

        return nll_test_mean

    def get_loss(self):
        total_likelihood = torch.zeros(1, dtype=self.params.torch_d_type)
        total_exp = torch.zeros(self.y.shape[0], dtype=self.params.torch_d_type)

        for covariate_name, cov in self.covariates.items():
            ll, e_arg = cov.loss()

            total_likelihood += self.y @ ll
            total_exp += e_arg

        total_exp = (total_exp + self.baseline * torch.ones(self.y.shape[0], dtype=self.params.torch_d_type))
        total_likelihood = total_likelihood + self.y @ (self.baseline * torch.ones(self.y.shape[0], dtype=self.params.torch_d_type))
        loss = -1 * (total_likelihood - self.params.delta * torch.sum(torch.exp(total_exp)))
        loss = self.get_nats_per_bin(self.y, total_exp)
        return loss

    def get_test_loss(self):
        total_likelihood = torch.zeros(1, dtype=self.params.torch_d_type)
        total_exp = torch.zeros(self.y_test.shape[0], dtype=self.params.torch_d_type)

        for covariate_name, cov in self.covariates.items():
            ll, e_arg = cov.test_loss()

            total_likelihood += self.y_test @ ll
            total_exp += e_arg

        total_exp = (total_exp + self.baseline * torch.ones(self.y_test.shape[0], dtype=self.params.torch_d_type))
        total_likelihood = total_likelihood + self.y_test @ (self.baseline * torch.ones(self.y_test.shape[0], dtype=self.params.torch_d_type))
        loss = -1 * (total_likelihood - self.params.delta * torch.sum(torch.exp(total_exp)))
        loss = self.get_nats_per_bin(self.y_test, total_exp)
        return loss

    def plot_covariates(self, evolution_df_dict, timer_obj):
        timer_obj.time_waste_start()


        with torch.no_grad():
            nll = self.get_loss()
            nll_test = self.get_test_loss()
            plt.style.use("ggplot")
            fig, axs = plt.subplots(2, 3, figsize=(3*len(self.covariates.keys()), 10))
            axs = axs.flatten()

            for dx, (name, covariate) in enumerate(self.covariates.items()):
                if name == 'History':
                    axs[dx].set_ylim([-7, 2])
                plot_mean, plot_std, plot_time, entire_mean, entire_std, entire_time = self.covariates[name].get_values_to_plot()

                plot_time, plot_mean, plot_std = zip(*sorted(zip(plot_time, plot_mean, plot_std)))
                plot_time = np.array(plot_time)
                plot_mean = np.array(plot_mean)
                plot_std = np.array(plot_std)

                axs[dx].plot(plot_time, plot_mean, label='posterior mean', color='tomato')
                axs[dx].fill_between(plot_time, plot_mean - 2 * plot_std, plot_mean + 2 * plot_std, alpha=0.3, color='salmon')

                if not covariate.etc_params['use_basis_form']:
                    axs[dx].plot(self.covariates[name].time.time_dict_t['u']().data.detach().numpy(),
                                np.zeros(self.covariates[name].time.time_dict['u'].shape[0]),
                                'o', color='orange', label='inducing points')
                axs[dx].set_title(name)
                axs[dx].legend()

                ev_dx = evolution_df_dict[name].shape[0]
                evolution_df_dict[name].at[ev_dx, 'plot_mean'] = np.copy(plot_mean)
                evolution_df_dict[name].at[ev_dx, 'plot_2std'] = np.copy(2 * plot_std)
                evolution_df_dict[name].at[ev_dx, 'plot_time'] = np.copy(plot_time)
                evolution_df_dict[name].at[ev_dx, 'entire_mean'] = np.copy(entire_mean)
                evolution_df_dict[name].at[ev_dx, 'entire_2std'] = np.copy(2 * entire_std)
                evolution_df_dict[name].at[ev_dx, 'entire_time'] = np.copy(entire_time)
                evolution_df_dict[name].at[ev_dx, 'nll'] = np.copy(nll.data.detach().numpy())
                evolution_df_dict[name].at[ev_dx, 'nll_test'] = np.copy(nll_test.data.detach().numpy())

                timer_obj.time_waste_end()
                evolution_df_dict[name].at[ev_dx, 'time_so_far'] = timer_obj.get_time()
                timer_obj.time_waste_start()
                evolution_df_dict[name].to_pickle(f'{self.params.gp_ev_path}_{name}')

            plt.subplots_adjust(hspace=1.0)
            plt.savefig(self.params.gp_filter_plot_path, dpi=300)
            print(f'nll: {nll_test.data.detach().numpy()}')
            plt.show()
            timer_obj.time_waste_end()





