import torch
import torch.fft
import numpy as np
import torch.nn.functional as F
from fft_conv import fft_conv, FFTConv1d


def next_fast_len(n, factors=[2, 3, 5, 7]):
    '''
      Returns the minimum integer not smaller than n that can
      be written as a product (possibly with repettitions) of
      the given factors.
    '''
    best = float('inf')
    stack = [1]
    while len(stack):
        a = stack.pop()
        if a >= n:
            if a < best:
                best = a;
                if best == n:
                    break; # no reason to keep searching
        else:
            for p in factors:
                b = a * p;
                if b < best:
                    stack.append(b)
    return best

def torch_xcorr(BATCH=1, signal_length=36000, device='cpu', factors=[2,3,5], dtype=torch.float):
    signal_length=36000
    # torch.rand is random in the range (0, 1)
    signal_1 = 1 - 2*torch.rand((BATCH, signal_length), device=device, dtype=dtype)
    signal_2 = 1 - 2*torch.rand((BATCH, signal_length), device=device, dtype=dtype)

    # just make the cross correlation more interesting
    signal_2 += 0.1 * signal_1;

    # output target length of crosscorrelation
    x_cor_sig_length = signal_length*2 - 1

    # get optimized array length for fft computation
    fast_length = next_fast_len(x_cor_sig_length, [2, 3])

    # the last signal_ndim axes (1,2 or 3) will be transformed
    fft_1 = torch.fft.rfft(signal_1, fast_length, dim=-1)
    fft_2 = torch.fft.rfft(signal_2, fast_length, dim=-1)

    # take the complex conjugate of one of the spectrums. Which one you choose depends on domain specific conventions

    fft_multiplied = torch.conj(fft_1) * fft_2

    # back to time domain.
    prelim_correlation = torch.fft.irfft(fft_multiplied, dim=-1)

    # shift the signal to make it look like a proper crosscorrelation,
    # and transform the output to be purely real

    final_result = torch.roll(prelim_correlation, (fast_length//2,))
    return final_result, torch.sum(final_result);


def torch_conv(signal, filter, offset):
    n = signal.shape[0] + filter.shape[0] - 1

    if offset > 0:
        left_pad = offset
        right_pad = 0
        start_dex = 0
        end_dex = signal.shape[0]

    elif offset < 0:
        left_pad = 0
        right_pad = -1 * offset
        start_dex = -1 * offset
        end_dex = start_dex + signal.shape[0]

    else:
        left_pad = 0
        right_pad = 0
        start_dex = 0
        end_dex = signal.shape[0]

    fft_signal = torch.fft.rfft(signal, n)
    fft_filter = torch.fft.rfft(filter, n)
    fft_multiplied = fft_signal * fft_filter

    ifft = torch.fft.irfft(fft_multiplied)
    ifft = F.pad(ifft, (int(left_pad), int(right_pad)))

    return ifft[start_dex: end_dex]


def main():
    x = torch.tensor([1.0, 2.0, 3.0])
    filter = torch.nn.Parameter(x)

    y = torch.tensor([4.0, 5.0, 6.0, 7.0])
    signal = torch.nn.Parameter(y)
    offset = 0

    prelim_correlation = torch_conv(signal, filter, offset)
    print(prelim_correlation)

if __name__ == '__main__':
    main()