import torch
import torch.nn.functional as F


def torch_batch_convolve(signal, filter, offset):
    '''

    :param signal: dim = batch_size x signal_length
    :param filter: dim = filter_length
    :param offset:
    :return:
    '''
    n = signal.shape[1] + filter.shape[1] - 1

    if offset > 0:
        left_pad = offset
        right_pad = 0
        start_dex = 0
        end_dex = signal.shape[1]

    elif offset < 0:
        left_pad = 0
        right_pad = -1 * offset
        start_dex = -1 * offset
        end_dex = start_dex + signal.shape[1]

    else:
        left_pad = 0
        right_pad = 0
        start_dex = 0
        end_dex = signal.shape[1]

    fft_signal = torch.fft.rfft(signal, n)
    fft_filter = torch.fft.rfft(filter, n)
    fft_multiplied = fft_signal * fft_filter

    ifft = torch.fft.irfft(fft_multiplied)
    ifft = F.pad(ifft, (int(left_pad), int(right_pad)))

    return ifft[start_dex: end_dex]
