import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import sqrtm

from GLM.GLM_Model.PoissonVariational import PoissonVariational
from GLM.GLM_Model.PoissonMAP import PoissonMAP
from GLM.GLM_Model import GLM_Model_GP, GLM_Model_MAP, GP_Covariate, MAP_Covariate
from Utils import utils


class Model_Runner:
    def __init__(self, params):
        self.params = params
        self.poisson_model = None
        self.variational_model = None
        self.map_model = None
        self.ml_model = None
        self.data_df = None

        self.hist_data = None
        self.stim_data = None

    def initialize_design_matrices(self):

        self.data_df = pd.read_pickle(self.params.expt_problem_data_path)

        self.hist_data = self.data_df.loc['History', 'data']
        self.spike_data = self.data_df.loc['History', 'data']
        self.stim1_data = self.data_df.loc['Stim1', 'data']
        self.stim2_data = self.data_df.loc['Stim2', 'data']
        self.stim3_data = self.data_df.loc['Stim3', 'data']
        self.hist_data_b = self.data_df.loc['Coupling_b', 'data']
        self.hist_data_c = self.data_df.loc['Coupling_c', 'data']

    def create_variational_covariates(self):
        kernel_prep_dict = {'chol': ['Kuu'], 'inv': ['Kuu']}

        # create glm object
        glm_gp = GLM_Model_GP.GLM_Model_GP(self.params)
        glm_gp.add_spike_history(self.spike_data)

        # history filter parameters
        hist_etc_params = {'use_exp_mean': True,
                           'use_basis_form': False}

        hist_bounds = {'m': [-np.inf, np.inf],
                       'r': [-np.inf, np.inf],
                       'u': [0, 0.25],
                       'alpha': [100, 100000],
                       'gamma': [100, 100000],
                       'sigma': [0.1, 15],
                       'kernel_epsilon_noise_std': [1e-4, 5],
                       'gain': [-15, -3],
                       'tau': [1e-4, 3e-3]
                       }

        hist_time_params = {'filter_offset': 1,
                            'filter_duration': 110,
                            'time_plot_min': 1,
                            'time_plot_max': 115,
                            'inducing_pt_spacing_init': 2,
                            'is_hist': True}

        hist_gp_params = {'alpha': [750.0, True],
                          'gamma': [1000.0, True],
                          'sigma': [np.sqrt(4), True],
                          'gain': [-5, False],
                          'tau': [1e-3, False],
                          'kernel_epsilon_noise_std': [1e-3, False],
                          'kernel_fn': [utils.decay_kernel_torch, False]}

        hist = GP_Covariate.GP_Covariate(self.params, hist_etc_params, self.hist_data,
                                         name='History',
                                         is_cov=False,
                                         use_bases=False)

        hist.add_bounds_params(hist_bounds)
        hist.add_gp_params(hist_gp_params)
        hist.add_time(hist_time_params)
        glm_gp.add_covariate(hist)

        ##########################################
        # Stim1
        ##########################################
        cue_etc_params = {'use_exp_mean': False,
                           'use_basis_form': False}

        cue_bounds = {'m': [-np.inf, np.inf],
                       'r': [-np.inf, np.inf],
                       'u': [0, 1.1],
                       'alpha': [50, 5000],
                       'gamma': [10, 3000],
                       'sigma': [0.1, 15],
                       'b': [300e-3, 800e-3],
                       'kernel_epsilon_noise_std': [1e-4, 1.5]
                       }

        cue_time_params = {'filter_offset': 0,
                            'filter_duration': 1000,
                            'time_plot_min': 0,
                            'time_plot_max': 1100,
                            'inducing_pt_spacing_init': 15,
                            'is_hist': False}

        cue_gp_params = {'alpha': [100.0, True], 'gamma': [600.0, True], 'sigma': [np.sqrt(4), True],
                          'b': [500e-3, True],
                          'kernel_epsilon_noise_std': [1e-3, False],
                          'kernel_fn': [utils.decay_kernel_torch, False]}

        cue = GP_Covariate.GP_Covariate(self.params, cue_etc_params, self.stim1_data, name='Stim1', is_cov=True)
        cue.add_bounds_params(cue_bounds)
        cue.add_gp_params(cue_gp_params)
        cue.add_time(cue_time_params)
        glm_gp.add_covariate(cue)

        ##########################################
        # Stim2
        ##########################################
        cue_etc_params = {'use_exp_mean': False,
                           'use_basis_form': False}

        cue_bounds = {'m': [-np.inf, np.inf],
                       'r': [-np.inf, np.inf],
                       'u': [-300e-3, 300e-3],
                       'alpha': [50, 5000],
                       'gamma': [10, 3000],
                       'sigma': [0.1, 15],
                       'b': [0e-3, 200e-3],
                       'kernel_epsilon_noise_std': [1e-4, 1.5]
                       }

        cue_time_params = {'filter_offset': -250,
                            'filter_duration': 500,
                            'time_plot_min': -300,
                            'time_plot_max': 300,
                            'inducing_pt_spacing_init': 15,
                            'is_hist': False}

        cue_gp_params = {'alpha': [100.0, True], 'gamma': [600.0, True], 'sigma': [np.sqrt(4), True],
                          'b': [100e-3, True],
                          'kernel_epsilon_noise_std': [1e-3, False],
                          'kernel_fn': [utils.decay_kernel_torch, False]}

        cue = GP_Covariate.GP_Covariate(self.params, cue_etc_params, self.stim2_data, name='Stim2', is_cov=True)
        cue.add_bounds_params(cue_bounds)
        cue.add_gp_params(cue_gp_params)
        cue.add_time(cue_time_params)
        glm_gp.add_covariate(cue)

        ##########################################
        # Stim1
        ##########################################
        cue_etc_params = {'use_exp_mean': False,
                           'use_basis_form': False}

        cue_bounds = {'m': [-np.inf, np.inf],
                       'r': [-np.inf, np.inf],
                       'u': [0, 500e-3],
                       'alpha': [50, 5000],
                       'gamma': [10, 3000],
                       'sigma': [0.1, 15],
                       'b': [100e-3, 500e-3],
                       'kernel_epsilon_noise_std': [1e-4, 1.5]
                       }

        cue_time_params = {'filter_offset': 0,
                            'filter_duration': 400,
                            'time_plot_min': 0,
                            'time_plot_max': 500,
                            'inducing_pt_spacing_init': 15,
                            'is_hist': False}

        cue_gp_params = {'alpha': [100.0, True], 'gamma': [600.0, True], 'sigma': [np.sqrt(4), True],
                          'b': [250e-3, True],
                          'kernel_epsilon_noise_std': [1e-3, False],
                          'kernel_fn': [utils.decay_kernel_torch, False]}

        cue = GP_Covariate.GP_Covariate(self.params, cue_etc_params, self.stim3_data, name='Stim3', is_cov=True)
        cue.add_bounds_params(cue_bounds)
        cue.add_gp_params(cue_gp_params)
        cue.add_time(cue_time_params)
        glm_gp.add_covariate(cue)

        ##########################################
        # Coupling a
        ##########################################
        cue_etc_params = {'use_exp_mean': False,
                          'use_basis_form': False}

        cue_bounds = {'m': [-np.inf, np.inf],
                      'r': [-np.inf, np.inf],
                      'u': [0, 200],
                      'alpha': [50, 10000],
                      'gamma': [10, 5000],
                      'sigma': [0.1, 15],
                      'b': [5e-3, 100e-3],
                      'kernel_epsilon_noise_std': [1e-4, 1.5]
                      }

        cue_time_params = {'filter_offset': 1,
                           'filter_duration': 100,
                           'time_plot_min': 0,
                           'time_plot_max': 150,
                           'inducing_pt_spacing_init': 2,
                           'is_hist': False}

        cue_gp_params = {'alpha': [5000.0, True], 'gamma': [1000.0, True], 'sigma': [np.sqrt(9), True],
                         'b': [15e-3, True],
                         'kernel_epsilon_noise_std': [1e-3, False],
                         'kernel_fn': [utils.decay_kernel_torch, False]}

        cue = GP_Covariate.GP_Covariate(self.params, cue_etc_params, self.hist_data_b, name='Coupling_b', is_cov=True)
        cue.add_bounds_params(cue_bounds)
        cue.add_gp_params(cue_gp_params)
        cue.add_time(cue_time_params)
        glm_gp.add_covariate(cue)

        ##########################################
        # Coupling b
        ##########################################
        cue_etc_params = {'use_exp_mean': False,
                          'use_basis_form': False}

        cue_bounds = {'m': [-np.inf, np.inf],
                      'r': [-np.inf, np.inf],
                      'u': [0, 200],
                      'alpha': [50, 10000],
                      'gamma': [10, 5000],
                      'sigma': [0.1, 15],
                      'b': [5e-3, 100e-3],
                      'kernel_epsilon_noise_std': [1e-4, 1.5]
                      }

        cue_time_params = {'filter_offset': 1,
                           'filter_duration': 100,
                           'time_plot_min': 0,
                           'time_plot_max': 150,
                           'inducing_pt_spacing_init': 2,
                           'is_hist': False}

        cue_gp_params = {'alpha': [5000.0, True], 'gamma': [1000.0, True], 'sigma': [np.sqrt(9), True],
                         'b': [30e-3, True],
                         'kernel_epsilon_noise_std': [1e-3, False],
                         'kernel_fn': [utils.decay_kernel_torch, False]}

        cue = GP_Covariate.GP_Covariate(self.params, cue_etc_params, self.hist_data_c, name='Coupling_c', is_cov=True)
        cue.add_bounds_params(cue_bounds)
        cue.add_gp_params(cue_gp_params)
        cue.add_time(cue_time_params)
        glm_gp.add_covariate(cue)

        self.variational_model = PoissonVariational(self.params, self.data_df, glm_gp, kernel_prep_dict)
        self.variational_model.initialize_variational_model()

    def create_map_covariates(self):
        glm_map = GLM_Model_MAP.GLM_Model_MAP(self.params)
        glm_map.add_spike_history(self.spike_data)

        ###################################
        # History
        ###################################
        hist_bounds = {'m': [-np.inf, np.inf],
                       'r': [0, np.inf]}

        hist_bases_params = {'bases_fn': utils.create_nonlinear_raised_cos,
                             'duration': 150,
                             'num_bases': 20,
                             'bin_size': self.params.delta,
                             'start_point': 0,
                             'end_point': 60e-3,
                             'nl_offset': 1e-4,
                             'offset': 1,
                             'filter_duration': self.params.duration_hist,
                             'filter_offset': 1,
                             'time_plot_min': 1,
                             'time_plot_max': 100}

        # hist = MAP_Covariate.MAP_Covariate(self.params, y, name='History', is_cov=False, is_hist=True)
        hist = MAP_Covariate.MAP_Covariate(self.params, self.hist_data, name='History', is_cov=False, is_hist=True)
        hist.add_bounds_params(hist_bounds)
        hist.add_bases_params(hist_bases_params)
        glm_map.add_covariate(hist)

        # stimulus 1 parameters
        cue_bounds = {'m': [-np.inf, np.inf],
                      'r': [0, 5]}

        cue_bases_params = {'bases_fn': utils.create_nonlinear_raised_cos,
                              'num_bases': 15,
                              'duration': 1000,  # self.params.duration_cov,
                              'bin_size': self.params.delta,
                              'end_point': 600e-3,
                              'start_point': 0,
                              'nl_offset': 1.3e-2,
                              'offset': 0,
                              'filter_duration': 1500,
                              'filter_offset': 0,
                              'time_plot_min': 0,
                              'time_plot_max': 1500}

        cue = MAP_Covariate.MAP_Covariate(self.params, self.stim1_data, name='Stim1', is_cov=True, is_hist=False)
        cue.add_bounds_params(cue_bounds)
        cue.add_bases_params(cue_bases_params)
        glm_map.add_covariate(cue)

        ######################
        # Lick Init
        ######################
        lick_init_bounds = {'m': [-np.inf, np.inf],
                            'r': [0, 2]}

        lick_init_bases_params = {'bases_fn': utils.create_raised_cosine_basis,
                                  'duration': 500,
                                  'num_bases': 20,
                                  'bin_size': self.params.delta,
                                  'start_point': 1,
                                  'end_point': 60e-3,
                                  'nl_offset': 1e-4,
                                  'offset': -250,
                                  'filter_duration': 500,
                                  'filter_offset': -250,
                                  'time_plot_min': -250,
                                  'time_plot_max': 250}

        lick_init = MAP_Covariate.MAP_Covariate(self.params, self.stim2_data, name='Stim2', is_cov=True, is_hist=False)
        lick_init.add_bounds_params(lick_init_bounds)
        lick_init.add_bases_params(lick_init_bases_params)
        glm_map.add_covariate(lick_init)

        ###################
        # Lick Train
        ###################
        lick_train_bounds = {'m': [-np.inf, np.inf],
                             'r': [0, 2]}

        lick_train_bases_params = {'bases_fn': utils.create_raised_cosine_basis,
                                  'duration': 500,
                                  'num_bases': 20,
                                  'bin_size': self.params.delta,
                                  'start_point': 1,
                                  'end_point': 60e-3,
                                  'nl_offset': 1e-4,
                                  'offset': 0,
                                  'filter_duration': 500,
                                  'filter_offset': 0,
                                  'time_plot_min': 0,
                                  'time_plot_max': 500}

        lick_train = MAP_Covariate.MAP_Covariate(self.params, self.stim3_data, name='Stim3', is_cov=True, is_hist=False)
        lick_train.add_bounds_params(lick_train_bounds)
        lick_train.add_bases_params(lick_train_bases_params)
        glm_map.add_covariate(lick_train)

        ###################################
        # Coupling a
        ###################################
        hist_bounds = {'m': [-np.inf, np.inf],
                       'r': [0, 2]}

        hist_bases_params = {'bases_fn': utils.create_raised_cosine_basis,
                             'duration': 125,
                             'num_bases': 20,
                             'bin_size': self.params.delta,
                             'start_point': 0,
                             'end_point': 60e-3,
                             'nl_offset': 1e-4,
                             'offset': 1,
                             'filter_duration': 125,
                             'filter_offset': 1,
                             'time_plot_min': 1,
                             'time_plot_max': 125}

        # hist = MAP_Covariate.MAP_Covariate(self.params, y, name='History', is_cov=False, is_hist=True)
        hist = MAP_Covariate.MAP_Covariate(self.params, self.hist_data_b, name='Coupling_b', is_cov=False, is_hist=False)
        hist.add_bounds_params(hist_bounds)
        hist.add_bases_params(hist_bases_params)
        glm_map.add_covariate(hist)

        ###################################
        # Coupling b
        ###################################
        hist_bounds = {'m': [-np.inf, np.inf],
                       'r': [0, 2]}

        hist_bases_params = {'bases_fn': utils.create_raised_cosine_basis,
                             'duration': 125,
                             'num_bases': 20,
                             'bin_size': self.params.delta,
                             'start_point': 0,
                             'end_point': 60e-3,
                             'nl_offset': 1e-4,
                             'offset': 1,
                             'filter_duration': 125,
                             'filter_offset': 1,
                             'time_plot_min': 1,
                             'time_plot_max': 125}

        # hist = MAP_Covariate.MAP_Covariate(self.params, y, name='History', is_cov=False, is_hist=True)
        hist = MAP_Covariate.MAP_Covariate(self.params, self.hist_data_c, name='Coupling_c', is_cov=False, is_hist=False)
        hist.add_bounds_params(hist_bounds)
        hist.add_bases_params(hist_bases_params)
        glm_map.add_covariate(hist)



        self.map_model = PoissonMAP(self.params, self.data_df, glm_map)
        self.map_model.initialize_model()

    def create_ml_covariates(self):
        glm_map = GLM_Model_MAP.GLM_Model_MAP(self.params)

        # stimulus 1 parameters
        stim1_bounds = {'m': [-np.inf, np.inf]}

        stim1_bases_params = {'bases_fn': utils.create_raised_cosine_basis,
                              'num_bases': 10,
                              'duration': self.params.duration_cov,
                              'bin_size': self.params.delta,
                              'end_point': 125e-3,
                              'start_point': 0,
                              'nl_offset': 2e-3,
                              'offset': self.params.offset_cov,
                              'filter_duration': self.params.duration_cov,
                              'filter_offset': self.params.offset_cov}

        stim1 = ML_Covariate.ML_Covariate(self.params, self.stim_data, name='Stimuli_1', is_cov=True, is_hist=False)
        stim1.add_bounds_params(stim1_bounds)
        stim1.add_bases_params(stim1_bases_params)
        glm_map.add_covariate(stim1)

        # history filter parameters
        hist_bounds = {'m': [-np.inf, np.inf]}

        hist_bases_params = {'bases_fn': utils.create_nonlinear_raised_cos,
                             'duration': 80,
                             'num_bases': 15,
                             'bin_size': self.params.delta,
                             'start_point': 0,
                             'end_point': 35e-3,
                             'nl_offset': 1e-4,
                             'offset': 1,
                             'filter_duration': self.params.duration_hist,
                             'filter_offset': self.params.offset_hist}

        hist = ML_Covariate.ML_Covariate(self.params, self.hist_data, name='History', is_cov=False, is_hist=True)
        hist.add_bounds_params(hist_bounds)
        hist.add_bases_params(hist_bases_params)
        glm_map.add_covariate(hist)

        self.ml_model = PoissonMAP(self.params, self.data_df, glm_map, self.params.run_toy_problem)
        self.ml_model.initialize_model()

    def train_variational(self):
        self.variational_model.train_variational_parameters()

    def train_map(self):
        self.map_model.train_map_parameters()

    def train_ml(self):
        self.ml_model.train_ml_parameters()


    def _add_training_params(self):
        pass

    def train_model(self, model='variational'):
        trained_params = self.poisson_model.train_variational()

    def _get_ml_h_k_mu(self):
        optimizer = Optimizer.Optimizer(self.params.gp_ml_opt, self.h.shape[0], b1=self.params.gp_ml_b1,
                                        b2=self.params.gp_ml_b2, step_size=self.params.gp_ml_step_size)

        for i in range(self.params.gp_ml_iter):
            grad = self.X.T @ self.y - self.params.delta * self.X.T @ np.exp(self.X @ self.h + self.Y @ self.k)
            update = optimizer.get_update(grad)
            self.h = self.h + update                                         # maximizing maximum likelihood

        plt.plot(self.h, label='ml')
        plt.plot(self.h_true, label='ground truth')
        plt.title('ml estimate')
        plt.show()

    def plot_updates(self):
        fig, axs = plt.subplots(self.h_evolution.shape[0] - 1, figsize=(10,60))
        fig.suptitle('GP Filter Evolution', y=0.92)

        for dx, (row, series) in enumerate(self.h_evolution.iloc[1:,:].iterrows()):
            axs[dx].plot(series['filter'], label='gp', color='k')
            axs[dx].plot(self.h_true, label='true', color='r')

            axs[dx].fill_between(np.arange(series['filter'].shape[0]), series['filter'] - series['cov'],
                                 series['filter'] + series['cov'], alpha=0.3, color='k')

            axs[dx].plot(series['inducing'], np.zeros(series['inducing'].shape[0]), 'o', color='orange', label='inducing points')
            axs[dx].legend()

            self._set_filter_axs(axs[dx])
            axs[dx].set_title(row)

        plt.subplots_adjust(hspace=0.3)
        fig.savefig('glm_data/gp_filter_evolution.pdf', dpi=300)
        plt.show()

    def _set_filter_axs(self, axs):
        len = self.h_time.shape[0]

        axs.set_xticks([i for i in np.arange(len + 1) if i % 50 == 0])
        labels = [int(i * self.params.time_div) for i in self.h_time if (i * self.params.time_div) % 50 == 0]
        labels.append(int(len / 2))
        axs.set_xticklabels(labels)








# def fn_min(half_coeff, x, y, Kinv):
#     temp = np.zeros(Kinv.shape[0])
#     return -1 * (h.T @ (x.T @ y) - np.sum(np.exp(x @ h)) - 0.5 * h.T @ (Kinv @ h))
#
# def jac(half_coeff, X, y, Kinv):
#     return X.T @ np.exp(X@h) - X.T @ y + Kinv @ h
#
# def hess(half_coeff, x, y, Kinv):
#     return x.T @ np.diag(np.exp(x@h)) @ x
def callback(h_updated):
    print('entered callback')
    mult_exp_mat = np.load('mult_exp_mat.npy')
    unused_exp = mult_exp_mat @ h_updated
    cov = np.load('callback_var.npy')

    # h_unused = np.random.multivariate_normal(unused_exp, cov)
    h_unused = unused_exp
    np.save('h_unused.npy', h_unused)

def fn_min(half_coeff, time, use_dx, unuse_dx, X, x_use, y, Kinv):
    delta = 0.001
    h = np.zeros(time.shape[0])
    h[use_dx] = half_coeff
    h[unuse_dx] = np.load('h_unused.npy')

    obj = -1 * (h.T @ (X.T @ y) - delta*np.sum(np.exp(X @ h)) - 0.5 * half_coeff @ (Kinv @ half_coeff)) + time.shape[0] * np.log(delta)
    print(obj)
    return obj

def jac(half_coeff, time, use_dx, unuse_dx, X, x_use, y, Kinv):
    delta = 0.001
    h = np.zeros(time.shape[0])
    h[use_dx] = half_coeff
    h[unuse_dx] = np.load('h_unused.npy')
    return delta*x_use.T @ np.exp(X@h) - x_use.T @ y + Kinv @ half_coeff

def hess(half_coeff, time, use_dx, unuse_dx, X, x_use, y, Kinv):
    delta = 0.001
    h = np.zeros(time.shape[0])
    h[use_dx] = half_coeff
    h[unuse_dx] = np.load('h_unused.npy')

    hess =  delta*x_use.T @ np.diag(np.exp(X@h)) @ x_use

    # if not utils.isPD(hess):
    #     return utils.nearestPD(hess)
    return hess
#
# u = 4
# unused_dx = [i for i in range(self.h_time.shape[0]) if i % u == 0]
# used_dx = [i for i in range(self.h_time.shape[0]) if i % u != 0]
# unused_time = self.h_time[unused_dx]
# hh_time = self.h_time[used_dx]
#
# h = self.h_true + 0.01*np.random.randn(self.h.shape[0])
# hh = h[used_dx]
# xx = self.X[:, used_dx]
# # kk = utils.decay_kernel(self.h_time.reshape(-1, 1),self.h_time.reshape(-1, 1), sigma_h=self.sigma_true,
# #                              alpha=self.alpha_true, gamma=self.gamma_true)[:,used_dx][used_dx,:]
# kk = ka.RBF(1).__call__(1000*hh_time.reshape(-1,1))
# kk_inv = np.linalg.inv(kk)
#
#
#
#
# # k_used_used = utils.decay_kernel(hh_time.reshape(-1, 1),hh_time.reshape(-1, 1), sigma_h=self.sigma_true,
# #                              alpha=self.alpha_true, gamma=self.gamma_true)
# k_used_used = ka.RBF(1).__call__(hh_time.reshape(-1,1)*1000, 1000*hh_time.reshape(-1,1))
# k_unused_used = ka.RBF(1).__call__(1000*unused_time.reshape(-1, 1),1000*hh_time.reshape(-1, 1))
# k_used_unused = ka.RBF(1).__call__(1000*hh_time.reshape(-1, 1), 1000*unused_time.reshape(-1, 1))
# k_unused_unused = ka.RBF(1).__call__(1000*unused_time.reshape(-1, 1), 1000*unused_time.reshape(-1, 1))

# k_unused_used = utils.decay_kernel(unused_time.reshape(-1, 1),hh_time.reshape(-1, 1), sigma_h=self.sigma_true,
#                              alpha=self.alpha_true, gamma=self.gamma_true)
# k_used_unused = utils.decay_kernel(hh_time.reshape(-1, 1),unused_time.reshape(-1, 1), sigma_h=self.sigma_true,
#                              alpha=self.alpha_true, gamma=self.gamma_true)
# k_unused_unused = utils.decay_kernel(unused_time.reshape(-1, 1),unused_time.reshape(-1, 1), sigma_h=self.sigma_true,
#                              alpha=self.alpha_true, gamma=self.gamma_true)

# time, use_dx, unuse_dx, X, x_use, y, Kinv
#
#
# u = 3
# r = 10
# h = np.copy(self.h_true) + 0.5*np.random.randn(self.h_true.shape[0])
# unuse_dx = [i for i in range(self.h_time.shape[0]) if i % u == 0]
# use_dx = [i for i in range(self.h_time.shape[0]) if i % u != 0]
# time = self.h_time
# hh_time = time[use_dx]
# unused_time = time[unuse_dx]
#
# K = utils.decay_kernel(hh_time.reshape(-1, 1),hh_time.reshape(-1, 1), sigma_h=self.sigma_true,
#                              alpha=self.alpha_true, gamma=self.gamma_true, noise_std=self.params.gp_noise_std)
# k_unused_used = utils.decay_kernel(unused_time.reshape(-1, 1),hh_time.reshape(-1, 1), sigma_h=self.sigma_true,
#                              alpha=self.alpha_true, gamma=self.gamma_true)
# k_used_unused = utils.decay_kernel(hh_time.reshape(-1, 1),unused_time.reshape(-1, 1), sigma_h=self.sigma_true,
#                              alpha=self.alpha_true, gamma=self.gamma_true)
# k_unused_unused = utils.decay_kernel(unused_time.reshape(-1, 1),unused_time.reshape(-1, 1), sigma_h=self.sigma_true,
#                              alpha=self.alpha_true, gamma=self.gamma_true, noise_std=self.params.gp_noise_std)
#
#
# X = np.copy(self.X)
# x_use = X[:,use_dx]
# y = np.copy(self.y)
# # y[y>0] = 1
#
# Kinv = scipy.linalg.inv(K)
# h_use = h[use_dx]
#
# mult_exp_mat = k_unused_used @ Kinv
# h_unuse_est = mult_exp_mat @ h_use
# np.save('mult_exp_mat.npy', mult_exp_mat)
#
# cov = k_unused_unused - k_unused_used @ Kinv @ k_unused_used.T
# np.save('callback_var.npy', cov)
# h_unused = np.random.multivariate_normal(mult_exp_mat @ h_use, cov)
# np.save('h_unused.npy', h_unused)
#
# a = scipy.optimize.minimize(fn_min, h_use, args=(time, use_dx, unuse_dx, X, x_use, y, Kinv),
#                         method='Newton-CG', jac=jac, hess=hess, options={'xtol':1e-4, 'disp':True, 'maxiter':100000},
#                             callback=callback)
# min_h_use = a.x
# mult_exp_mat = k_unused_used @ Kinv
# h_unuse_est = mult_exp_mat @ min_h_use
# h_use_est = k_used_unused @ (np.linalg.inv(k_unused_unused) @ h_unuse_est)
#
# estimated_h_all = np.zeros(self.h_true.shape[0])
# estimated_h_all[use_dx] = h_use_est
# estimated_h_all[unuse_dx] = h_unuse_est
# plt.plot(estimated_h_all)
# plt.plot(self.h_true)
# plt.show()






# k_used_used = ka.RBF(r).__call__(hh_time.reshape(-1,1)*1000, 1000*hh_time.reshape(-1,1))
# k_unused_used = ka.RBF(r).__call__(1000*unused_time.reshape(-1, 1),1000*hh_time.reshape(-1, 1))
# k_used_unused = ka.RBF(r).__call__(1000*hh_time.reshape(-1, 1), 1000*unused_time.reshape(-1, 1))
# k_unused_unused = ka.RBF(r).__call__(1000*unused_time.reshape(-1, 1), 1000*unused_time.reshape(-1, 1))


#
# a = scipy.optimize.minimize(fn_min, hh, args=(xx, self.y, kk_inv),
#                         method='Newton-CG', jac=jac, hess=hess, options={'xtol':1e-5, 'disp':True, 'maxiter':100000},
#                             callback=callback)
# h_used = a.x
# # h_unused = -1*(k_unused_used) @ (utils.nearestPD(np.linalg.inv(k_used_used)) @ h_used)
# h_unused = self.h_true[unused_dx]
# h_all = np.zeros(self.h_time.shape[0])
# h_all[used_dx] = h_used
# h_all[unused_dx] = h_unused
# plt.plot(h_all)
# plt.plot(self.h_true)
# plt.show()
#
#
# plt.plot(h_used)
# plt.plot(self.h_true[used_dx])
# plt.show()
#
# k_unused_unused = utils.decay_kernel(10*unused_time.reshape(-1, 1),10*unused_time.reshape(-1, 1), sigma_h=1000,
#                              alpha=0.1, gamma=0.45)
#
# sample = np.random.multivariate_normal(np.zeros(k_unused_unused.shape[0]), k_unused_unused)
# plt.plot(sample)
# plt.show()
#
# def obj_fn(alpha, X, y, h, h_time):
#     K = utils.decay_kernel(h_time.reshape(-1, 1), h_time.reshape(-1, 1), sigma_h=2, alpha=alpha, gamma=600, noise_std=1)
#     Kinv = np.linalg.inv(K)
#     W = X.T @ np.diag(np.exp(X @ h)) @ X
#     Wsqrt = np.sqrt(1e-3) * sqrtm(X.T @ (np.diag(np.exp(X @ h)) @ X))
#     I = np.identity(K.shape[0])
#     obj1 = y.T @ X @ h
#     obj2 = -1 * np.sum(np.exp(X@h))
#     obj3 = -0.5 * h.T @ Kinv @ h
#     obj4 = 0 #-0.5 * np.linalg.slogdet(I + Wsqrt @ (K @ Wsqrt))
#
#     return -1*(obj1 + obj2 + obj3 + obj4)
#
# def obj_grad(alpha, X, y, h, h_time):
#     K = utils.decay_kernel(h_time.reshape(-1, 1), h_time.reshape(-1, 1), sigma_h=2,
#                            alpha=alpha, gamma=600, noise_std=0.001)
#     Kinv = np.linalg.inv(K)
#     Wsqrt = np.sqrt(1e-3) * sqrtm(X.T @ (np.diag(np.exp(X @ h)) @ X))
#     I = np.identity(K.shape[0])
#
#     K_prime = K * -1 * np.log(np.outer(np.exp(h_time ** 2), np.exp(h_time ** 2)))
#     term1 = 0.5 * h.T @ (Kinv @ (K_prime @ Kinv)) @ h
#     term2 = -0.5 * np.trace(np.linalg.inv(I + Wsqrt @ K @ Wsqrt) @ (Wsqrt @ K_prime @ Wsqrt))
#
#     return -1*(term1 + term2)
#
# def callback(alpha):
#     print(f'alpha: {alpha}')
#
# a = optimize.minimize(obj_fn, 500, args=(x_design, y_true, h_true, h_time),
#                                 method='BFGS', jac=obj_grad, options={'xtol': 1e-4, 'disp': True, 'maxiter': 500},
#                                 callback=callback)
#
# def test_deriv():
#     a = 1
#     b = 1
#     c = 3.1
#     d = 1.3
#     e = 2.4
#     time = np.array([1,2]).reshape(-1,1)
#     time_add_sq = -1 * np.log(np.outer(np.exp(time ** 2), np.exp(time ** 2)))
#     A = np.array([[a, 0], [0, b]])
#     B = np.array([[c, d], [0, e]])
#
#     K = utils.decay_kernel(time, time, sigma_h=2,
#                            alpha=2, gamma=1)
#
#     inside_log = a*e*K[1,1] + a*b + c*e*K[0,0]*K[1,1] + c*b*K[0,0] + d*e*K[1,0]*K[1,1] + b*d*K[1,0]
#     deriv = (-a*e*(2**2 + 2**2)*K[1,1] - c*e*K[0,0]*(2**2 + 2**2)*K[1,1] - c*e*K[1,1]*(1**2 + 1**2)*K[0,0] -
#              b*c*(1**2 + 1**2)*K[0,0] - d*e*K[1,0]*(2**2 + 2**2)*K[1,1] - d*e*K[1,1]*(1**2 + 2**2)*K[1,0] - d*b*(1**2 + 2**2)*K[1,0])
#
#     K_prime = K * time_add_sq
#     grad1 = deriv/inside_log
#     grad2 = np.trace(np.linalg.inv(A + B@K) @ B@K_prime)
def _log_likelihood_brute( params, *args):
    alpha = params[0]
    gamma = params[1]
    sigma = params[2]
    print(alpha)

    h = args[0]
    time = args[1]
    X = args[2]
    y = args[3]

    delta = self.params.delta
    Kinv = self.GP.K_all_inv

    obj = -1 * (h.T @ (X.T @ y) - delta * np.sum(np.exp(X @ h)) - 0.5 * h @ (Kinv @ h)) + \
          time.shape[0] * np.log(delta)

    return obj

# y = np.zeros(200)
# for alpha in range(y.shape[0]):
#     y[alpha] = _log_marginal_deriv_wrt_alpha_test(np.array([5*alpha]), self.h_true)

def _log_marginal_deriv_wrt_alpha_test(alpha_in):
    h = self.h
    alpha = alpha_in[0]
    sigma = self.sigma
    gamma = self.gamma

    K = utils.decay_kernel(self.h_time.reshape(-1, 1), self.h_time.reshape(-1, 1), sigma_h=sigma,
                           alpha=alpha, gamma=gamma, noise_std=0.001)
    Kinv = np.linalg.inv(K)
    W = self.params.delta * self.X.T @ (np.diag(np.exp(self.X @ self.h_true)) @ self.X)
    Winv = np.linalg.inv(W)
    Wsqrt = np.sqrt(self.params.delta) * sqrtm(self.X.T @ (np.diag(np.exp(self.X @ self.h_true)) @ self.X))
    I = np.identity(K.shape[0])
    K_prime = K * -1 * np.log(np.outer(np.exp(self.h_time ** 2), np.exp(self.h_time ** 2)))

    # term1 = 0.5 * self.h_true.T @ (Kinv @ (K_prime @ Kinv)) @ self.h_true
    # term2 = -0.5 * np.trace(np.linalg.inv(I + Wsqrt @ K @ Wsqrt) @ (Wsqrt @ K_prime @ Wsqrt))

    term1 = 0.5*h.T @ (Kinv @ K_prime @ Kinv) @ h - 0.5*np.trace(np.linalg.inv(Winv + K)@K_prime)
    term2 = np.zeros(self.X.shape[1])
    inv_K_inv_W = np.linalg.inv(Kinv + W)
    for i in range(self.X.shape[1]):
        term2[i] = -0.5*np.trace(inv_K_inv_W @ (self.X.T @ np.diag(self.X[:,i]*np.exp(self.X@h)) @ self.X))

    # X_reshape = self.X.reshape(self.X.shape[0], 1, -1)
    # inv_Kinv_pl_W = np.linalg.inv(Kinv + W)
    #
    # diag_mat = diag_3d(self.X*np.exp(self.X@h).reshape(-1,1))
    # part1 = np.einsum('ij,jkl->ikl', self.X.T, diag_mat)
    # part2 = np.einsum('ijl,jk->ikl', part1, self.X)
    # part3 = np.einsum('ij,jkl->ikl', inv_Kinv_pl_W, part2)
    # term2 = np.trace(part3)
    #
    # diag_mat = diag_3d(self.X[:,self.X.shape[1]//2:] * np.exp(self.X @ h).reshape(-1, 1))
    # part1 = np.einsum('ij,jkl->ikl', self.X.T, diag_mat)
    # part2 = np.einsum('ijl,jk->ikl', part1, self.X)
    # part3 = np.einsum('ij,jkl->ikl', inv_Kinv_pl_W, part2)
    # term2 = np.concatenate([term2, np.trace(part3)])

    term3 = np.linalg.inv(I + K@W) @ K_prime @ (self.X.T @ self.y - self.params.delta*self.X.T @ np.exp(self.X@h))
    grad = term1 + np.sum(term2*term3)

    return grad

def diag_3d(M):
    b = np.zeros((M.shape[0], M.shape[0], M.shape[1]))
    diag_2 = np.arange(M.shape[0])
    # b[diag_2, diag_2, :] = M

    for i in range(M.shape[1]):
        b[:,:,i] = M[:,i]
    return b