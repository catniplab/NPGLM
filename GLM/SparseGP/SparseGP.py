import torch

from collections import OrderedDict


class CovarianceMatrix:
    def __init__(self, kernel_fn):
        self.kernel_fn = kernel_fn
        self.K = None
        self.L = None
        self.inv = None

    def update_kernel(self, time_1, time_2, params, gp_params):
        self.K = self.kernel_fn(time_1, time_2, params, **gp_params)

    def set_cholesky(self):
        self.L = torch.cholesky(self.K, upper=True)

    def set_inverse(self):
        self.inv = torch.inverse(self.K)


class SparseGP:
    def __init__(self, params, time, gp_params):
        self.params = params
        self.gp_params = gp_params
        self.time = time

        self.Kxx = CovarianceMatrix(gp_params.gp_kernel[f'kernel_fn'])
        self.Kxu = CovarianceMatrix(gp_params.gp_kernel[f'kernel_fn'])
        self.Kuu = CovarianceMatrix(gp_params.gp_kernel[f'kernel_fn'])
        self.kernel_dict = OrderedDict({'Kxx': self.Kxx, 'Kuu': self.Kuu, 'Kxu': self.Kxu})

        self.Vuu = None
        self.Vnu = None

    def update_kernels(self):

        for name, kernel in self.kernel_dict.items():
            time_key_1 = name[1]
            time_key_2 = name[2]

            if time_key_1 == 'u':
                time_1 = self.time.time_dict_t['u']()
            else:
                time_1 = self.time.time_dict[time_key_1]

            if time_key_2 == 'u':
                time_2 = self.time.time_dict_t['u']()
            else:
                time_2 = self.time.time_dict[time_key_2]

            kernel.update_kernel(time_1,
                                 time_2,
                                 self.params,
                                 self.gp_params.gp_params_t)

    def compute_needed_chol_and_inv(self, chol_inv_update):
        chol_update = chol_inv_update['chol']
        inv_update = chol_inv_update['inv']

        for matrix in chol_update:
            self.__dict__[matrix].set_cholesky()

        for matrix in inv_update:
            self.__dict__[matrix].set_inverse()

    def cho_solve(self, cholesky_matrix_name, vec_solve):
        matrix_suffix = cholesky_matrix_name[1:]

        return torch.cholesky_solve(vec_solve.reshape(-1,1), self.kernel_dict[f'K{matrix_suffix}'].L, upper=True).reshape(-1)

    def __getitem__(self, item):
        if 'inv' in item:
            last_dx = item.find('_inv')
            return self.__dict__[item[:last_dx]].inv
        if item[0] == 'K':
            return self.__dict__[item].K
        elif item[0] == 'L':
            return self.__dict__['K' + item[1:]].L
