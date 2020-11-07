import torch
import numpy as np
import os

from GLM.NPGLM import NPGLM
from GLM.GLM_Model import GLM_Model_GP, GLM_Model_MAP, GP_Covariate, MAP_Covariate
from Utils import utils


def main():
    expt = 'expt_supp'
    params_path = 'GLM/GLM_Params/params.json'
    os.chdir('..')

    # set up an npglm object
    npglm = NPGLM.NPGLM(expt, params_path)
    npglm.initialize_parameters()
    npglm.run_initialization_scheme()
    npglm.run_initilization_npglm()

    ##########################################
    # History
    ##########################################
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

    npglm.add_covariate('History', hist_bounds, hist_gp_params, hist_time_params, hist_etc_params)

    ##########################################
    # Stim1
    ##########################################
    stim1_etc_params = {'use_exp_mean': False,
                        'use_basis_form': False}

    stim1_bounds = {'m': [-np.inf, np.inf],
                  'r': [-np.inf, np.inf],
                  'u': [0, 2.0],
                  'alpha': [50, 5000],
                  'gamma': [10, 3000],
                  'sigma': [0.1, 15],
                  'b': [300e-3, 800e-3],
                  'kernel_epsilon_noise_std': [1e-4, 1.5]
                  }

    stim1_time_params = {'filter_offset': 0,
                         'filter_duration': 1000,
                         'time_plot_min': 0,
                         'time_plot_max': 1100,
                         'inducing_pt_spacing_init': 15}

    stim1_gp_params = {'alpha': [100.0, True], 'gamma': [600.0, True], 'sigma': [np.sqrt(4), True],
                       'b': [500e-3, True],
                       'kernel_epsilon_noise_std': [1e-3, False],
                       'kernel_fn': [utils.decay_kernel_torch, False]}

    npglm.add_covariate('Stim1', stim1_bounds, stim1_gp_params, stim1_time_params, stim1_etc_params)

    ##########################################
    # Stim2
    ##########################################
    stim2_etc_params = {'use_exp_mean': False,
                        'use_basis_form': False}

    stim2_bounds = {'m': [-np.inf, np.inf],
                    'r': [-np.inf, np.inf],
                    'u': [-300e-3, 300e-3],
                    'alpha': [50, 5000],
                    'gamma': [10, 3000],
                    'sigma': [0.1, 15],
                    'b': [0e-3, 200e-3],
                    'kernel_epsilon_noise_std': [1e-4, 1.5]
                    }

    stim2_time_params = {'filter_offset': -250,
                         'filter_duration': 500,
                         'time_plot_min': -300,
                         'time_plot_max': 300,
                         'inducing_pt_spacing_init': 15}

    stim2_gp_params = {'alpha': [100.0, True], 'gamma': [600.0, True], 'sigma': [np.sqrt(4), True],
                       'b': [100e-3, True],
                       'kernel_epsilon_noise_std': [1e-3, False],
                       'kernel_fn': [utils.decay_kernel_torch, False]}

    npglm.add_covariate('Stim2', stim2_bounds, stim2_gp_params, stim2_time_params, stim2_etc_params)

    ##########################################
    # Stim3
    ##########################################
    stim3_etc_params = {'use_exp_mean': False,
                      'use_basis_form': False}

    stim3_bounds = {'m': [-np.inf, np.inf],
                  'r': [-np.inf, np.inf],
                  'u': [0, 500e-3],
                  'alpha': [50, 5000],
                  'gamma': [10, 3000],
                  'sigma': [0.1, 15],
                  'b': [100e-3, 500e-3],
                  'kernel_epsilon_noise_std': [1e-4, 1.5]
                  }

    stim3_time_params = {'filter_offset': 0,
                       'filter_duration': 400,
                       'time_plot_min': 0,
                       'time_plot_max': 500,
                       'inducing_pt_spacing_init': 15}

    stim3_gp_params = {'alpha': [100.0, True], 'gamma': [600.0, True], 'sigma': [np.sqrt(4), True],
                     'b': [250e-3, True],
                     'kernel_epsilon_noise_std': [1e-3, False],
                     'kernel_fn': [utils.decay_kernel_torch, False]}

    npglm.add_covariate('Stim3', stim3_bounds, stim3_gp_params, stim3_time_params, stim3_etc_params)

    ##########################################
    # Coupling b
    ##########################################
    couple_b_etc_params = {'use_exp_mean': False,
                      'use_basis_form': False}

    couple_b_bounds = {'m': [-np.inf, np.inf],
                  'r': [-np.inf, np.inf],
                  'u': [0, 200],
                  'alpha': [50, 10000],
                  'gamma': [10, 5000],
                  'sigma': [0.1, 15],
                  'b': [5e-3, 100e-3],
                  'kernel_epsilon_noise_std': [1e-4, 1.5]
                  }

    couple_b_time_params = {'filter_offset': 1,
                       'filter_duration': 100,
                       'time_plot_min': 0,
                       'time_plot_max': 150,
                       'inducing_pt_spacing_init': 2}

    couple_b_gp_params = {'alpha': [5000.0, True], 'gamma': [1000.0, True], 'sigma': [np.sqrt(9), True],
                     'b': [15e-3, True],
                     'kernel_epsilon_noise_std': [1e-3, False],
                     'kernel_fn': [utils.decay_kernel_torch, False]}

    npglm.add_covariate('Coupling_b', couple_b_bounds, couple_b_gp_params, couple_b_time_params, couple_b_etc_params)

    ##########################################
    # Coupling c
    ##########################################
    couple_c_etc_params = {'use_exp_mean': False,
                           'use_basis_form': False}

    couple_c_bounds = {'m': [-np.inf, np.inf],
                       'r': [-np.inf, np.inf],
                       'u': [0, 200],
                       'alpha': [50, 10000],
                       'gamma': [10, 5000],
                       'sigma': [0.1, 15],
                       'b': [5e-3, 100e-3],
                       'kernel_epsilon_noise_std': [1e-4, 1.5]
                       }

    couple_c_time_params = {'filter_offset': 1,
                            'filter_duration': 100,
                            'time_plot_min': 0,
                            'time_plot_max': 150,
                            'inducing_pt_spacing_init': 2}

    couple_c_gp_params = {'alpha': [5000.0, True], 'gamma': [1000.0, True], 'sigma': [np.sqrt(9), True],
                          'b': [30e-3, True],
                          'kernel_epsilon_noise_std': [1e-3, False],
                          'kernel_fn': [utils.decay_kernel_torch, False]}

    npglm.add_covariate('Coupling_c', couple_c_bounds, couple_c_gp_params, couple_c_time_params, couple_c_etc_params)

    # proceeding adding of covariate filter parameters train the model
    npglm.train_npglm()

if __name__ == '__main__':
    main()