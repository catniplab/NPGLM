import pandas as pd
import numpy as np
import torch

from collections import OrderedDict


class GLM_Model(torch.nn.Module):
    def __init__(self, params):
        super(GLM_Model, self).__init__()

        self.params = params
        self.delta = params.delta
        self.covariates = OrderedDict()
        self.baseline = None
        self.y = None

    def add_spike_history(self, y):
        y[:,0] = 0
        self.y = torch.flatten(torch.tensor(y[: -1 * self.params.num_test_trials, :], dtype=self.params.torch_d_type))
        self.y_test = torch.flatten(torch.tensor(y[-1 * self.params.num_test_trials:, : ], dtype=self.params.torch_d_type))

    def add_covariate(self, covariate):
        self.covariates[covariate.name] = covariate
        self.covariates[covariate.name].initialize_design_matrix()

    def initialize_covariates(self):
        '''
        initialize design matrices and initial values

        :return:
        '''

        if self.params.inference_type == 'basis':
            full_covariates = torch.ones((self.y.shape[0], 1), dtype=self.params.torch_d_type)

            for covariate_name, covariate in self.covariates.items():
                full_covariates = torch.cat([full_covariates, covariate.X], axis=1)

            h = torch.zeros(full_covariates.shape[1], dtype=self.params.torch_d_type, requires_grad=True)
            h.data[0] = torch.log(torch.sum(self.y) / (self.params.delta * (self.y.shape[0])))
            self.optimizer = torch.optim.LBFGS([h], lr=1, history_size=100, max_iter=30, line_search_fn='strong_wolfe')
            optimizer_closure = self.ml_closure(full_covariates, h)

            print(self.optimizer.step(optimizer_closure))

            h.requires_grad = False
            self.baseline = torch.nn.Parameter(torch.tensor(h[0], dtype=self.params.torch_d_type).reshape(1), requires_grad=True)
            beg_dex = 1

            for covariate_name, covariate in self.covariates.items():
                m_length = covariate.X.shape[1]
                covariate.initialize_filter_params(h[beg_dex: beg_dex + m_length])
                # covariate.initialize_bound_params()
                covariate.update_filter_params_with_transform(self)
                beg_dex += m_length

            self.register_map_params()

        elif self.params.inference_type == 'gp':

            for covariate_name, covariate in self.covariates.items():
                relevant_df = pd.read_pickle(f'{self.params.basis_ev_path}_{covariate_name}')
                self.baseline = torch.nn.Parameter(torch.tensor(np.array(relevant_df.loc[relevant_df.shape[0] - 1, 'baseline']), dtype=self.params.torch_d_type), requires_grad=True)
                covariate.update_gp_params_with_transform(self)
                covariate.initialize_filter_params(relevant_df, self)


        print('done')

    def set_training_parameters(self, param_list):
        self.training_parameters = []

        for param in self.named_parameters():
            if param[0] in param_list:
                self.training_parameters.append(param[1])

    def set_optimizing_parameters_grad(self, params_to_optimize):
        for name, param in self.named_parameters():
            if name not in params_to_optimize:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def ml_closure(self, full_covariates, h):

        def closure():
            self.optimizer.zero_grad()

            if self.params.likelihood_type == 'logistic':
                sig = torch.nn.Sigmoid()
                loss_fn = torch.nn.BCELoss(reduction='sum')
                loss = loss_fn(sig(full_covariates @ h), self.y) + 2 * h[1:].t() @ h[1:]

            elif self.params.likelihood_type == 'poisson':
                log_like = self.y.T @ full_covariates @ h - self.delta * torch.sum(torch.exp(full_covariates @ h)) - 2*h[1:].t() @ h[1:]
                loss = -1 * log_like

            loss.backward()

            return loss

        return closure
