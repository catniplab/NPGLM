# NPGLM
A library for inferring neural encoding relations when cast in the form of a generalized linear model.  Each of the "filters" has a GP prior placed over it.  Using the Sparse Variational Gaussian Process (SVGP) framework allows for nonparametric inference of the filters that is expressive and flexible.  Use of inducing points allows for reasonable run times and the ability to learn the appropriate filter length.

# Background
Encoding models of neural activity allow us to infer the firing rate as a function of relevant experimental variables.  
GLMs offer an attractive way to infer these functional relationships. For example, in an experiment with one relevant stimuli a simple generative model may be of the form

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;\mathbf{w}_1&space;&\sim&space;\mathcal{N}(0,&space;\mathbf{\Sigma}_1)&space;\\&space;k_1(t)&space;&=&space;\sum_i&space;\phi_i(t)&space;w_{1,&space;i}&space;\\&space;\lambda(t)&space;&=&space;s_1(t)&space;*&space;k_1(t)&space;\\&space;y(t)&space;\rvert&space;\lambda(t)&space;&\sim&space;\text{Poisson}(\lambda(t))&space;\\&space;\end{align*}&space;\newline&space;\begin{align*}&space;\mathbf{w}&space;&\triangleq&space;\text{``basis&space;weights''}&space;\\&space;\phi_i(t)&space;&\triangleq&space;\text{``basis&space;$i$&space;from&space;a&space;set&space;of&space;N&space;basis&space;used&space;for&space;filter&space;characteriziation''}\\&space;k_1(t)&space;&\triangleq&space;\text{``filter&space;for&space;stimuli&space;1''}&space;\\&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\begin{align*}&space;\mathbf{w}_1&space;&\sim&space;\mathcal{N}(0,&space;\mathbf{\Sigma}_1)&space;\\&space;k_1(t)&space;&=&space;\sum_i&space;\phi_i(t)&space;w_{1,&space;i}&space;\\&space;\lambda(t)&space;&=&space;s_1(t)&space;*&space;k_1(t)&space;\\&space;y(t)&space;\rvert&space;\lambda(t)&space;&\sim&space;\text{Poisson}(\lambda(t))&space;\\&space;\end{align*}&space;\newline&space;\begin{align*}&space;\mathbf{w}&space;&\triangleq&space;\text{``basis&space;weights''}&space;\\&space;\phi_i(t)&space;&\triangleq&space;\text{``basis&space;$i$&space;from&space;a&space;set&space;of&space;N&space;basis&space;used&space;for&space;filter&space;characteriziation''}\\&space;k_1(t)&space;&\triangleq&space;\text{``filter&space;for&space;stimuli&space;1''}&space;\\&space;\end{align*}" title="\begin{align*} \mathbf{w}_1 &\sim \mathcal{N}(0, \mathbf{\Sigma}_1) \\ k_1(t) &= \sum_i \phi_i(t) w_{1, i} \\ \lambda(t) &= s_1(t) * k_1(t) \\ y(t) \rvert \lambda(t) &\sim \text{Poisson}(\lambda(t)) \\ \end{align*} \newline \begin{align*} \mathbf{w} &\triangleq \text{``basis weights''} \\ \phi_i(t) &\triangleq \text{``basis $i$ from a set of N basis used for filter characteriziation''}\\ k_1(t) &\triangleq \text{``filter for stimuli 1''} \\ \end{align*}" /></a>

While simple, attractive, and widely used this characterization of the filters of interest in terms of a set of predefined basis functions limits the expressiveness and flexibility of the model.  

Our method tries to minimize the a priori assumptions and specifications that need to be made by taking advantage of the Gaussian Process (GP) framework. Arbitrary filters can be characterized and learned while the amount of fixed model parameters is limited only to the number of inducing points.

# Compatible Data Format
To use NPGLM please transform your data to pandas data frames with the following attributes:

| Index | 'Data' |
| --- | --- |
| 'covariate_1_name' | Trials x Trial_Length Matrix of Data |
| 'covariate_2_name' | Trials x Trial_Length Matrix of Data |
| 'covariate_3_name' | Trials x Trial_Length Matrix of Data |

For example, spiking history can be incorporated by having one of the indexed variables be "History"
with an appropriate data column containing the Trials x Trial_Length matrix of neural spiking history.

# Usage
```python
import NPGLM
import utils
import pickle
import numpy as np

# read in experimental .json params and create an NPGLM object
params = pickle.read('params_folder/params.json')
glm_gp = NPGLM(params)

# initialize the first covariate, this involves specifiying initial 
# parameters of the filter such as:
#       initial filter offset: the initial start time of the filter
#       filter_duration_init: the initial length of the filter
#       time_plot_min/time_plot_max: parameters for plotting the evolution over training
#       inducing_pt_spacing_init: the initial spacing of the inducing points
#       is_hist: whether or not this is the self spiking history filter

cov1_time_params = {'filter_offset': 1,
                    'filter_duration_init': 110,
                    'time_plot_min': 1,
                    'time_plot_max': 115,
                    'inducing_pt_spacing_init': 2,
                    'is_hist': False}

# if you wish to bound the hyperparameters, variational parameters, or inducing point locations
# these can be specified here
#       m: variational mean
#       r: cholesky factors of variational covariance
#       alpha: decay parameter of DSE kernel
#       gamma: inverse length scale parameter of DSE kernel
#       kernel_epsilon_noise_std: isotropic additive noise term of DSE kernel modification
cov1_params_bounds =  {'m': [-np.inf, np.inf],
                       'r': [-np.inf, np.inf],
                       'u': [0, 0.25],
                       'alpha': [100, 100000],
                       'gamma': [100, 100000],
                       'sigma': [0.1, 15],
                       'kernel_epsilon_noise_std': [1e-4, 5]}

# hyperparameters can be initialized to some values and if desired these can be fixed and 
# will remain unchanged during optimization
cov1_gp_params_init = {'alpha': [750.0, True], 
                       'gamma': [1000.0, True], 
                       'sigma': [2, True],
                       'kernel_epsilon_noise_std': [1e-3, False],
                       'kernel_fn': [utils.decay_kernel_torch, False]}

# create a GP_Covariate object with the specified parameters
cov1 = GP_Covariate(params, cov1_params, cov1_data, name='Covariate_1', is_cov=True)
cov1.add_bounds_params(cov1_params_bounds)
cov1.add_gp_params(cov1_gp_params_init)
cov1.add_time(cov1_params, glm_gp)

# add the covariate object to the model
npglm.add_covariate(hist)

# train NPGLM
npglm.train_variational()
```

You can add more covariates if you wish i.e. spiking history is a viable candidate as well as any other variables in the experiment.  All data will be saved to the folder specified in the .json parameters file.  For multiple experiments, we recommend using the .json file to specify different folders for different experimental conditions so that the whole process can be completely scripted.

# JSON Parameters
The file GLM/GLM_Params/params.json contains a list of parameters that can be taken advantage of for quick scripting of running an array of experiments. You can specify the training routine i.e. how many iterations are spent optimizing hyperparameters as well as variatonal parameters and the maximum hyperparameters are allowed to change per iteration. 

| Parameter | Description |
| --- | --- |
| gp_filter_plot_path | absolute path to save filter plot (showing mean/95% cred. interval/inducing points) |
| gp_ev_path | path to save data |
| time_optimization_radius | maximum filter expansion per iteration |
| delta | bin size |
| likelihood type | likelihood function of the observations, supports 'poisson'/'logistic' |
| inference type | use 'basis' or 'gp' to specify inference using basis functions / npglm |

# Demo
Demo/demo.py contains a quick demo of the script running.



