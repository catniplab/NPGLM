from GLM.GLM_Model.Model_Runner import Model_Runner
from Utils import utils, SpikeGen

import torch

def main():
    expt = 'expt_supp'
    params = utils.Params('GLM/GLM_Params/params.json')
    params.torch_d_type = torch.float64

    params.gp_filter_plot_path = f"Results_Data/{expt}/data/gp_filter_plot"
    params.basis_filter_plot_path = f"Results_Data/{expt}/data/basis_filter_plot"
    params.gp_ev_path = f"Results_Data/{expt}/data/neuron_gp_ev"
    params.basis_ev_path = f"Results_Data/{expt}/data/neuron_map_ev"

    params.inference_type = 'basis'
    gp = Model_Runner(params)
    gp.initialize_design_matrices()
    gp.create_map_covariates()
    gp.train_map()

    params.inference_type = 'gp'
    gp = Model_Runner(params)
    gp.initialize_design_matrices()
    gp.create_variational_covariates()
    gp.train_variational()

if __name__ == '__main__':
    main()