from collections import OrderedDict
from GLM.GLM_Model.BoundTransform import BoundTransform

class Covariate:
    def __init__(self, params, x, **property_dict):
        '''

        :param params:
        :param x: numpy array of covariate values
        :param property_dict:
        '''
        self.params = params
        self.delta = params.delta
        self.name = property_dict['name']
        self.is_cov = property_dict['is_cov']

        self.params_to_optimize = OrderedDict()
        self.init_bounds_params = None
        self.bounds_params = OrderedDict()
        self.bounds_transform = OrderedDict()
        self.filter_params = None
        self.time = None

        self.x = x
        self.X = None

    def add_bounds_params(self, bound_params):
        self.init_bounds_params = OrderedDict(bound_params)

    def initialize_design_matrix(self):
        raise NotImplementedError('Must be Implemented in Child Class')

    def initialize_bound_params(self):
        self.bounds_params = OrderedDict()
        self.bounds_transform = OrderedDict()

        for name, param_value in self.params_to_optimize.items():
            lower = self.init_bounds_params[name][0]# * np.array(np.ones(param_len))
            upper = self.init_bounds_params[name][1]# * np.array(np.ones(param_len))

            self.bounds_params[name] = [lower, upper]
            self.bounds_transform[name] = BoundTransform(self.params, [lower, upper])

    def update_filter_params_with_transform(self, glm_obj):
        for name, param in self.filter_params.filter_params.items():
            if param.requires_grad:
                glm_obj.register_parameter(name=f'{self.name}_{name}', param=self.filter_params.filter_params[name])

                lower = self.init_bounds_params[name][0]
                upper = self.init_bounds_params[name][1]

                self.bounds_params[name] = [lower, upper]
                self.bounds_transform[name] = BoundTransform(self.params, [lower, upper])
                self.filter_params.update_with_transform(self.bounds_transform[name], name)


    def update_params_to_optimize(self, params):
        '''
        Create dictionary containing all parameters that need be optimized, recall these will just be references to the
        original pytorch tensors

        :param params: params dict containing numpy/torch variants of the variables
        '''

        for param_name, param_value in params.items():
            if param_value.requires_grad:
                self.params_to_optimize[param_name] = param_value

    def get_log_likelihood_terms(self):
        raise NotImplementedError('Must be Implemented In Child Class')

