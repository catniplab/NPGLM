import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import sys
import numpy as np
from scipy.stats import norm


from Utils import utils


class SpikeGenerator:
    def __init__(self, params, trial_length, train_length, x_separation, hist_filter=None, stim_filter=None):
        self.params = params
        self.delta = params.delta
        self.hist_length = 100
        self.train_length = train_length
        self.trial_length = trial_length
        self.stim_filter_true = stim_filter
        self.hist_filter_true = hist_filter
        self.hist_filter_time = None
        self.stim_filter_time = None

        self.x_separation = x_separation
        self.x = None
        self.X = None
        self.Xh = None

        self.y = None
        self.Y_hist = None

        self.baseline = np.log(0.05 / 0.001)
        self.full_model_data_df = pd.DataFrame(columns=['is_covariate', 'is_spike_train', 'filter',
                                                        'data', 'test_data', 'filter_time', 'baseline']).rename_axis('name')

    def initialize_filters(self):
        if self.stim_filter_true is None:
            self.create_toy_stim_filter()

        if self.hist_filter_true is None:
            self.create_toy_hist_filter()

        self.x = np.zeros(self.trial_length)
        x_true_mask = np.arange(self.x_separation // 6, self.trial_length, self.x_separation // 3)
        self.x[x_true_mask] = 1

        self.X = utils.create_convolution_matrix(self.x, 0, self.stim_filter_true.shape[0])
        self.Xh = self.X @ self.stim_filter_true

        self.stim_filter_time = self.delta * np.arange(0, self.stim_filter_true.shape[0])
        self.hist_filter_time = self.delta * np.arange(self.hist_filter_true.shape[0])



    def create_toy_stim_filter(self):
        x = np.arange(0, 275)

        z = norm(loc=150, scale=25)
        z = z.pdf(x)

        zz = norm(loc=125, scale=25)
        zz = -0.6 * zz.pdf(x)
        z = z + zz

        z = (z / z.max()) * 2
        z = z[::2]

        self.stim_filter_true = z
        np.save('toy_problem_data/data/gaussian_filter.npy', z)

    def create_toy_hist_filter(self):
        hist_time = self.delta * np.arange(0, 100)
        self.hist_filter_true = -10 * np.exp(-hist_time / (self.params.tau))
        self.k_true_time = self.delta * np.arange(self.hist_filter_true.shape[0])

        a = -20 * np.exp(-hist_time / (self.params.tau / 30))
        a[10:] = a[10:] + 7 * np.exp(-hist_time[:-10] / (self.params.tau / 4))

        a[16:] = a[16:] + (-2) * np.exp(-hist_time[:-16] / (self.params.tau * 3))
        a = gaussian_filter1d(a, sigma=3)
        a = 0.8 * a

        self.hist_filter_true = a
        np.save('toy_problem_data/data/gaussian_filter.npy', a)

    def generate_spike_sequence(self):
        self.y = np.zeros(self.trial_length)

        for t in np.arange(1, self.trial_length):
            if t <= self.hist_filter_true.shape[0]:
                history_term = np.flip(self.y[0:t]) @ self.hist_filter_true[0:t]
            else:
                history_term = np.flip(self.y[t-self.hist_length:t]) @ self.hist_filter_true

            intensity = np.exp(self.Xh[t] + self.baseline + history_term)

            try:
                self.y[t] = np.random.poisson(self.delta * intensity)
            except:
                sys.exit('Readjust Filter Parameters -- Unstable bin generated')

    def plot(self):
        X = self.x.reshape(-1, self.trial_length//10)
        Y = self.y.reshape(-1, self.trial_length//10)

        spike_event_list = [np.where(Y[i, :] > 0)[0] for i in range(Y.shape[0])]
        cov_event_list = [np.where(X[i, :] > 0)[0] for i in range(X.shape[0])]

        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        ax1.plot(self.stim_filter_time, self.stim_filter_true, color='red', label='stimulus filter')
        ax1.set_title('Stimulus Filter')
        ax1.set_xlabel('time (s)')

        k_time = self.delta * np.arange(0, 100)
        ax2.plot(k_time, self.hist_filter_true[0:100], color='black', label='history filter')
        ax2.set_title('History Filter')
        ax2.set_xlabel('time (s)')

        ax3.eventplot(spike_event_list[0], color='black', label='spike')
        ax3.eventplot(spike_event_list[0:], color='black')
        ax3.eventplot(cov_event_list[0], linewidths=3, color='red', label='stimuli')
        ax3.eventplot(cov_event_list[0:], linewidths=3, color='red')
        ax3.set_xlabel('time (ms)')
        ax3.set_ylabel('trial')
        ax3.set_title('Raster Plot')

        plt.legend()
        plt.subplots_adjust(hspace=0.5)
        plt.subplots_adjust(wspace=0.3)
        fig.savefig('full_model_data/generated_data.pdf', dpi=300, bbox_inches='tight')
        fig.savefig('toy_problem_data/generated_data/generated_data_plot.pdf', dpi=300, bbox_inches='tight')
        plt.show()

    def save_data(self):
        self.full_model_data_df.loc['History', 'is_covariate'] = False
        self.full_model_data_df.loc['History', 'is_spike_train'] = True
        self.full_model_data_df.loc['History', 'data'] = self.y[0:self.train_length]
        self.full_model_data_df.loc['History', 'test_data'] = self.y[self.train_length:]
        self.full_model_data_df.loc['History', 'filter'] = self.hist_filter_true
        self.full_model_data_df.loc['History', 'filter_time'] = self.k_true_time
        self.full_model_data_df.loc['History', 'baseline'] = self.baseline

        self.full_model_data_df.loc['Stimuli_1', 'is_covariate'] = True
        self.full_model_data_df.loc['Stimuli_1', 'is_spike_train'] = False
        self.full_model_data_df.loc['Stimuli_1', 'data'] = self.x[0:self.train_length]
        self.full_model_data_df.loc['Stimuli_1', 'test_data'] = self.x[self.train_length:]
        self.full_model_data_df.loc['Stimuli_1', 'filter'] = self.stim_filter_true
        self.full_model_data_df.loc['Stimuli_1', 'filter_time'] = self.stim_filter_time
        print(f'{np.where(self.y[0:self.train_length] > 0)[0].shape[0]} spikes generated')

        self.full_model_data_df.to_pickle(f'{self.params.toy_problem_data_path}data_df')


