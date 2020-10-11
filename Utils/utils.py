import numpy as np
import scipy.linalg as la
import torch
import threading, time

import matplotlib.pyplot as plt
from scipy.special import kv
import json

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class ResumableTimer:
    def __init__(self):
        self.start_time = None
        self.time_waste_start_time = None
        self.wasted_time = 0

    def start(self):
        self.start_time = time.perf_counter()

    def time_waste_start(self):
        self.time_waste_start_time = time.perf_counter()

    def time_waste_end(self):
        self.wasted_time += (time.perf_counter() - self.time_waste_start_time)

    def get_time(self):
        return time.perf_counter() - (self.start_time + self.wasted_time)



def raised_cosine(x, dur):
    raised_cosine = np.multiply((np.abs(x / dur) < 0.5), (np.cos(x * 2 * np.pi / dur) * 0.5 + 0.5))
    return raised_cosine

def create_raised_cosine_basis(**kwargs):
    duration = kwargs['duration']
    num_bases = kwargs['num_bases']

    num_bins_per_basis = 1 if duration == 0 else np.ceil(duration / 1).astype(int)
    time_indices = np.tile(np.arange(1, num_bins_per_basis + 1), [num_bases, 1])

    db_center = num_bins_per_basis / (3 + num_bases)
    width = 4 * db_center
    bin_centers = 2 * db_center + db_center * np.arange(num_bases)
    centered_values = time_indices - bin_centers.reshape([bin_centers.shape[0], 1])
    binned_values = raised_cosine(centered_values, width)

    return binned_values

def create_nonlinear_raised_cos(**kwargs):
    num_bases = kwargs['num_bases']
    bin_size = kwargs['bin_size']
    start_point = kwargs['start_point']
    end_point = kwargs['end_point']
    nl_offset = kwargs['nl_offset']

    def ff(x, dc):
        # values greater than pi -> pi
        x = x * (np.pi * (1 / dc) * (1 / 2))
        x[x > np.pi] = np.pi
        x[x < -np.pi] = -1 * np.pi
        x = (np.cos(x) + 1) / 2
        return x

    nlin = lambda x: np.log(x + 1e-20)
    invnl = lambda x: np.exp(x - 1e-20)

    end_points = np.array([start_point, end_point])
    nl_offset = nl_offset

    y_range = nlin(end_points + nl_offset)
    db = np.diff(y_range) / (num_bases - 1)
    centers = np.arange(y_range[0], y_range[1] + db, db).reshape([num_bases, 1])
    max_time_bin = invnl(y_range[1] + 2 * db) - nl_offset
    iht = np.arange(0, max_time_bin + bin_size, bin_size)
    iht_rep = np.tile(nlin(iht + nl_offset), [num_bases, 1])
    centers_rep = np.tile(centers, [1, iht.shape[0]])
    ih_basis = ff(iht_rep - centers_rep, db)

    # self.bases_series = ih_basis
    return ih_basis

def create_design_matrix(x, filter_len, filter_offset):
    x_design = np.zeros((x.shape[0], filter_len))
    input_len = x.shape[0]
    pre_coverage_dx = filter_len - (-1 * filter_offset) - 1
    post_coverage_counter = 1

    for i in range(x_design.shape[0]):

        if i <= pre_coverage_dx:
            x_design[i, 0: (i + 1) + -1 * filter_offset] = x[i + -1 * filter_offset:: -1]

        elif i > pre_coverage_dx and i + -1 * filter_offset <= x.shape[0] - 1:
            upper_dx = i + -1 * filter_offset
            lower_dx = upper_dx - filter_len
            x_design[i, :] = x[upper_dx: lower_dx: -1]

        else:
            x_design[i, post_coverage_counter:] = x[input_len - 1: input_len - filter_len + (
                        post_coverage_counter - 1): -1]
            post_coverage_counter += 1

    return x_design

def update_design_matrix(x, X, time_obj, mult_factor=2, return_torch=False, maximum=25):
    diff_upper = 2*int(np.ceil((time_obj.time_dict['numpy']['u_h'].max() - time_obj.time_dict['numpy']['x_h'].max()) / time_obj.delta))
    diff_lower = 2*int(np.ceil((time_obj.time_dict['numpy']['x_h'].min() - time_obj.time_dict['numpy']['u_h'].min()) / time_obj.delta))

    num_add_upper = diff_upper if diff_upper > 0 else 0
    num_add_lower = diff_lower if diff_lower > 0 else 0

    if num_add_upper > num_add_lower:
        num_add_lower = num_add_upper

    elif num_add_upper < num_add_lower:
        num_add_upper = num_add_lower

    new_offset = time_obj.offset - num_add_lower
    new_duration = time_obj.duration + num_add_lower + num_add_upper

    if num_add_upper == 0 and num_add_lower == 0:
        return X

    upper_times = time_obj.time_dict['numpy']['x_h'].max() + time_obj.delta * np.arange(1, num_add_upper + 1)
    lower_times = time_obj.time_dict['numpy']['x_h'].min() - time_obj.delta * np.arange(1, num_add_lower + 1)
    new_times = np.concatenate([lower_times, upper_times])

    X_new = create_convolution_matrix(x, new_offset, new_duration)
    time_obj.update_tracker_add_design_points(new_times)
    time_obj.offset = new_offset
    time_obj.duration = new_duration

    return X_new

def create_convolution_matrix(x, new_offset, new_duration):
    if new_offset <= 0:
        if np.abs(new_offset) < np.abs(new_duration):
            front_zeros = np.zeros(np.abs(new_duration + new_offset - 1))
            rear_zeros = np.zeros(np.abs(new_offset))
            x_padded = np.concatenate([front_zeros, x, rear_zeros])
            first_row = np.flip(x_padded[0:new_duration])
            first_column = x_padded[new_duration-1: x.shape[0] + new_duration - 1]

        elif np.abs(new_offset) > np.abs(new_duration):
            rear_zeros = np.zeros(np.abs(new_offset + new_duration) + new_duration)
            x_padded = np.concatenate([x, rear_zeros])
            first_row = np.flip(x_padded[np.abs(new_offset+new_duration) + 1: np.abs(new_offset+new_duration) + 1 + new_duration])
            first_column = x_padded[np.abs(new_offset + new_duration) + 1 + new_duration - 1:
                                    np.abs(new_offset + new_duration) + 1 + new_duration - 1 + x.shape[0]]

    elif new_offset > 0:
        zeros_to_pad = np.zeros(new_offset + new_duration - 1)
        x_padded = np.concatenate([zeros_to_pad, x])
        first_row = np.flip(x_padded[0:new_duration])
        first_column = x_padded[new_duration - 1: x.shape[0] + new_duration - 1]

    X_new = la.toeplitz(first_column, first_row)

    return X_new

def create_convolution_matrix_torch(x, new_offset, new_duration):
    if new_offset <= 0:
        if np.abs(new_offset) < np.abs(new_duration):
            front_zeros = np.zeros(np.abs(new_duration + new_offset - 1))
            rear_zeros = np.zeros(np.abs(new_offset))
            x_padded = np.concatenate([front_zeros, x, rear_zeros])
            first_row = np.flip(x_padded[0:new_duration])
            first_column = x_padded[new_duration-1: x.shape[0] + new_duration - 1]

        elif np.abs(new_offset) > np.abs(new_duration):
            rear_zeros = np.zeros(np.abs(new_offset + new_duration) + new_duration)
            x_padded = np.concatenate([x, rear_zeros])
            first_row = np.flip(x_padded[np.abs(new_offset+new_duration) + 1: np.abs(new_offset+new_duration) + 1 + new_duration])
            first_column = x_padded[np.abs(new_offset + new_duration) + 1 + new_duration - 1:
                                    np.abs(new_offset + new_duration) + 1 + new_duration - 1 + x.shape[0]]

    elif new_offset > 0:
        zeros_to_pad = np.zeros(new_offset + new_duration - 1)
        x_padded = np.concatenate([zeros_to_pad, x])
        first_row = np.flip(x_padded[0:new_duration])
        first_column = x_padded[new_duration - 1: x.shape[0] + new_duration - 1]

    X_new = la.toeplitz(first_column, first_row)

    return X_new


def decay_kernel(X1_p, X2_p, **kwargs):
    X1 = X1_p.reshape(-1,1)
    X2 = X2_p.reshape(-1,1)

    kwarg_keys = kwargs.keys()

    for key in kwarg_keys:
        if 'alpha' in key:
            alpha_key = key
        if 'gamma' in key:
            gamma_key = key
        if 'sigma' in key:
            sigma_key = key

    sigma_h = kwargs[sigma_key]
    alpha = kwargs[alpha_key]
    gamma = kwargs[gamma_key]
    noise_std = kwargs['kernel_epsilon_noise_std']

    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    decaying_matrix = np.sum(np.exp(-1*alpha*(X1 ** 2)), 1).reshape(-1, 1) * np.sum(np.exp(-1*alpha*X2 ** 2),1)

    if X1.shape != X2.shape:
        return np.exp(-1 * gamma * sqdist) * decaying_matrix * sigma_h ** 2
    else:
        if noise_std.shape[0] == 1:
            return np.exp(-1 * gamma * sqdist) * decaying_matrix * sigma_h ** 2 + noise_std*np.identity(X1.shape[0])
        else:
            return np.exp(-1 * gamma * sqdist) * decaying_matrix * sigma_h ** 2 + torch.diag(noise_std)

def decay_kernel_torch(X1_p, X2_p, params, **kwargs):

    kwarg_keys = kwargs.keys()

    for key in kwarg_keys:
        if 'alpha' in key:
            alpha_key = key
        if 'gamma' in key:
            gamma_key = key
        if 'sigma' in key:
            sigma_key = key

    if 'b' not in kwarg_keys:
        b_key = lambda : torch.tensor(np.array([0.0]), dtype=params.torch_d_type)

    else:
        b_key = kwargs['b']

    sigma_h = kwargs[sigma_key]
    alpha = kwargs[alpha_key]
    gamma = kwargs[gamma_key]

    noise_std = kwargs['kernel_epsilon_noise_std']

    X1 = X1_p.reshape(-1, 1) - b_key()
    X2 = X2_p.reshape(-1, 1) - b_key()

    sqdist = torch.sum(X1 ** 2, 1).reshape(-1,1) + torch.sum(X2 ** 2, 1) - 2 * torch.ger(X1.reshape(-1), X2.reshape(-1))
    decaying_matrix = torch.sum(torch.exp(-1*alpha()*(X1 ** 2)), 1).reshape(-1, 1) * torch.sum(torch.exp(-1*alpha()*X2 ** 2), 1)
    # decaying_matrix = np.exp(-1 * alpha * X1.T ** 2) * np.exp(-1 * alpha * X2 ** 2).reshape(-1, 1)

    if X1.shape != X2.shape:
        return torch.exp(-1 * gamma() * sqdist) * decaying_matrix * sigma_h() ** 2
    else:
        # if noise_std.shape[0] == 1:
        return torch.exp(-1 * gamma() * sqdist) * decaying_matrix * sigma_h() ** 2 + noise_std()*torch.eye(X1.shape[0], dtype=params.torch_d_type)
        # else:
        #     return torch.exp(-1 * gamma * sqdist) * decaying_matrix * sigma_h ** 2 + torch.diag(noise_std)

def decay_ou_kernel_torch(X1_p, X2_p, params, **kwargs):

    kwarg_keys = kwargs.keys()

    for key in kwarg_keys:
        if 'alpha' in key:
            alpha_key = key
        if 'gamma' in key:
            gamma_key = key
        if 'sigma' in key:
            sigma_key = key
        if 'ouconst' in key:
            ou_const_key = key

    if 'b' not in kwarg_keys:
        b_key = torch.tensor(np.array([0.0]), dtype=params.torch_d_type)

    else:
        b_key = kwargs['b']()

    sigma_h = kwargs[sigma_key]()
    alpha = kwargs[alpha_key]()
    gamma = kwargs[gamma_key]()
    ou_const = kwargs[ou_const_key]()

    noise_std = kwargs['kernel_epsilon_noise_std']()

    X1 = X1_p.reshape(-1, 1) - b_key
    X2 = X2_p.reshape(-1, 1) - b_key

    sqdist = torch.sum(X1 ** 2, 1).reshape(-1,1) + torch.sum(X2 ** 2, 1) - 2 * torch.ger(X1.reshape(-1), X2.reshape(-1))
    diff = torch.abs(X1 - X2.reshape(1,-1))
    decaying_matrix = torch.sum(torch.exp(-1*alpha*(X1 ** 2)), 1).reshape(-1, 1) * torch.sum(torch.exp(-1*alpha*X2 ** 2), 1)
    # decaying_matrix = np.exp(-1 * alpha * X1.T ** 2) * np.exp(-1 * alpha * X2 ** 2).reshape(-1, 1)

    if X1.shape != X2.shape:
        return 1 * torch.exp(-1 * ou_const * diff)*torch.exp(-300*X1**2 - 300*X2.reshape(1, -1)**2) \
               + torch.exp(-1 * gamma * sqdist) * decaying_matrix * sigma_h ** 2
    else:
        # if noise_std.shape[0] == 1:
        return 1 * torch.exp(-1 * ou_const * diff)*torch.exp(-300*X1**2 - 300*X2.reshape(1, -1)**2)\
               + torch.exp(-1 * gamma * sqdist) * decaying_matrix * sigma_h ** 2 + noise_std[0]*torch.eye(X1.shape[0], dtype=params.torch_d_type)
        # else:
        #     return torch.exp(-1 * gamma * sqdist) * decaying_matrix * sigma_h ** 2 + torch.diag(noise_std)


def decay_kernel_torch_hist(X1_p, X2_p, **kwargs):

    kwarg_keys = kwargs.keys()

    for key in kwarg_keys:
        if 'alpha' in key:
            alpha_key = key
        if 'gamma' in key:
            gamma_key = key
        if 'sigma' in key:
            sigma_key = key

        if 'alparam2' in key:
            alpha2_key = key
        if 'gaparam2' in key:
            gamma2_key = key

    if 'b' not in kwarg_keys:
        b_key = torch.tensor(np.array([0.0]), dtype=torch.double)

    else:
        b_key = kwargs['b']

    sigma_h = kwargs[sigma_key]
    alpha = kwargs[alpha_key]
    gamma = kwargs[gamma_key]
    #
    # alpha2 = kwargs[alpha2_key]
    gamma2 = kwargs[gamma2_key]

    noise_std = kwargs['kernel_epsilon_noise_std']

    X1 = X1_p.reshape(-1, 1) - b_key
    X2 = X2_p.reshape(-1, 1) - b_key

    sqdist = torch.sum(X1 ** 2, 1).reshape(-1,1) + torch.sum(X2 ** 2, 1) - 2 * torch.ger(X1.reshape(-1), X2.reshape(-1))
    # sqdist = torch.log(torch.abs(X1 - X2.reshape(1,-1)) + 1)
    decaying_matrix = torch.sum(torch.exp(-1*alpha*(X1 ** 2)), 1).reshape(-1, 1) * torch.sum(torch.exp(-1*alpha*X2 ** 2), 1)
    # decaying_matrix_p = torch.sum(torch.exp(-1*kwargs['ap']*(X1 ** 2)), 1).reshape(-1, 1) * torch.sum(torch.exp(-1*kwargs['ap']*X2 ** 2), 1)
    # decaying_matrix = np.exp(-1 * alpha * X1.T ** 2) * np.exp(-1 * alpha * X2 ** 2).reshape(-1, 1)
    # return torch.exp(-1 * gamma * sqdist)
    if X1.shape != X2.shape:
        K1 = torch.exp(-1 * gamma * sqdist) * decaying_matrix * sigma_h ** 2
    else:
        K1 = torch.exp(-1 * gamma * sqdist) * decaying_matrix * sigma_h ** 2 + noise_std[0]*torch.eye(X1.shape[0], dtype=torch.double)

    decaying_matrix2 = torch.sum(torch.exp(-1 * (alpha/10) * (X1 ** 2)), 1).reshape(-1, 1) * torch.sum(torch.exp(-1 * (alpha/10) * X2 ** 2), 1)
    K3 = (torch.ones(K1.shape[0], K1.shape[1], dtype=torch.double) - decaying_matrix)\
         * torch.exp(-1 * gamma2 * sqdist) * decaying_matrix2 * sigma_h ** 2
    # K2 = torch.exp(-1 * gamma2 * sqdist) * decaying_matrix2 * sigma_h ** 2

    return K1

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3

def adam_update(grad, m, v, b1, b2, step_size, dx):
    eps = 10e-12
    m = (1 - b1) * grad + b1 * m  # First  moment estimate.
    v = (1 - b2) * (grad ** 2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1 ** (dx + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (dx + 1))
    dx += 1

    return step_size * mhat / (np.sqrt(vhat) + eps), m, v, dx


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def matrix_valid(A):
    is_pd = isPD(A)
    is_symm = check_symmetric(A)

    if is_pd and is_symm:
        return True

    else:
        return False