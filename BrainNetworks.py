from copy import deepcopy
import pickle as pk
import h5py
from scipy.integrate import odeint
import scipy as sp
import numpy as np
from features import synchronizationLikelihood
import cmath
import matplotlib.pyplot as plt
from scipy.signal.windows import blackmanharris
from parabolic import parabolic
from scipy.signal import butter,lfilter
import AnalysisAndPlots as aap

def Kura(init, t, A, w_nat, a):
    '''
    This function returns the theta-dots for each oscillator. It is mainly just an auxiliary function to be called by
    the scipy odeint integrator

    :param init: vector of initial phases
    :param t: unimportant; only here to satisfy odeint convention
    :param A: connectivity matrix
    :param w_nat: vector of natural frequencies
    :param a: coupling strength
    :return: vector of theta-dots
    '''
    theta = np.array(init)
    delta_theta = np.subtract.outer(theta, theta)
    dot_theta = w_nat + a * np.einsum('ij,ji->i', A, np.sin(delta_theta))
    return dot_theta

def runK(A, phases0, w_nat, alpha, time):
    '''
    :param A: connectivity matrix
    :param phases0: vector of initial phases
    :param w_nat: vector of natural frequencies
    :param time: vector of timepoints to run the dynamics on. pass in something like np.linspace(0,100)
    :param alpha: coupling constant
    :return: returns the phases over all time points, shape (N,T).
    '''
    result = odeint(Kura, phases0, time, args=(A, w_nat, alpha)).T
    order_params = orderParameter(result)
    order_params = np.array([abs(complex_OP2(result[:,i])) for i in range(len(result[0]))])
    return result, order_params

def complex_OP2(theta):
    '''
    :param theta: vector of phases
    :return: complex order parameter. Take abs() to get R.
    '''
    return (1.0 / len(theta)) * sum(cmath.exp(complex(0, a)) for a in theta)

def orderParameter(theta):
    '''
    :param theta: (N,T) matrix showing the phases of all oscillators over time.
    :return: (T,) vector of magnitudes of the complex order parameter for all times
    '''
    N, _ = theta.shape
    return (1.0/N)*np.abs(np.sum(np.exp(theta *1j), axis = 0))

def averagedOP(start, end, OPData):
    return np.sum(OPData[start:end]) / (end - start)

def runSim(A, phases, frequencies, alpha, dt, steps):
    time = np.linspace(0, dt * steps, steps)
    thetas, order_params = runK(A, phases, frequencies, alpha, time)
    return thetas, order_params

def freq_from_fft(sig, fs): #source: https://gist.github.com/jgomezdans/434642/8b738ff4bff7f549f6255f5da3668f910b7352a2
    """
    Estimate frequency from peak of FFT
    """
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = np.fft.rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = np.argmax(abs(f))  # Just use this for less-accurate, naive version
    true_i = parabolic(np.log(abs(f)), i)[0]
    # Convert to equivalent frequency
    return fs * true_i / len(windowed)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def get_laplacian(A):
    result = -deepcopy(A)
    for i in range(len(result[0])):
        result[i][i] += np.sum(A[i])
    return result

def synchronizability(A):
    L = get_laplacian(A)
    for i in range(len(A)):
        A[i][i] = 1
    evals, v = sp.linalg.eigh(L)
    return evals[1]/evals[-1]

def get_timescale_max_data(A_list, nat_freqs, alpha, dt, steps):
    syncs, maxes, means, times = [], [], [], []
    for i in range(len(A_list)):
        init_phases = np.random.rand(len(nat_freqs)) * 2 * np.pi - np.pi #randomly choose initial phases
        thetas, order_params = runSim(A_list[i], init_phases, nat_freqs, alpha, dt, steps)
        sync = synchronizability(A_list[i])
        time_to_max = np.argmax(order_params) / steps
        max = np.max(order_params)
        syncs.append(sync)
        means.append(np.mean(order_params))
        times.append(time_to_max)
        maxes.append(max)
    return np.array(syncs), np.array(maxes), np.array(means), np.array(times)


def preprocessed_tseries(series, fs, band = [15, 30]):
    clean_tseries = np.array(series)
    clean_tseries -= np.mean(clean_tseries, axis = 0)
    if band is not None:
        clean_tseries = butter_bandpass_filter(clean_tseries, band[0], band[1], fs)
    return clean_tseries

def get_tseries_and_fs(f, band = [15,30], series_idx = None):
    '''
    Can be used for the full ecog data dataSets_clean.mat or an individual SEEG file like HUP131-short-ictal-block-1.mat.
    :param f: the h5py file object
    :param band: the desired band (for ex: use [15,30] for beta band. If no specific band, set to None
    :param series_idx: Set series_idx to the desired index if used for the full ecog data (otherwise leave as None)
    :return: (N,T) processed timeseries, as well as sampling frequency Fs
    '''
    if series_idx is not None:      #assuming full ecog data file
        refs = f['dataSets_clean']['data']
        refs_sample_freq = f['dataSets_clean']['Fs']
        fs = int(f[refs_sample_freq[series_idx][0]][()][0][0])

        tseries = f[refs[series_idx][0]][()].T
        tseries = preprocessed_tseries(tseries, fs, band = band)
        return tseries, fs
    else:       #assuming an individual SEEG file
        fs = int(f['Fs'][()][0][0])
        tseries = f['evData'][()].T
        tseries = preprocessed_tseries(tseries, fs, band=band)
    return tseries, fs


def construct_sync_likelihood_nets(f, band = [15, 30], pRef = 0.05, series_idx = None, name = ""):
    '''
    Can be used for the full ecog data dataSets_clean.mat or an individual SEEG file like HUP131-short-ictal-block-1.mat.
    Constructs networks for all 1-second intervals, and saves the result to a pickle file. To load, use something like
    A_list = pk.load(open("sync_likelihood_net_0.pk", "rb"))
    :param f: the h5py file object
    :param band: the desired band (for ex: use [15,30] for beta band. If no specific band, set to None
    :param pRef: pRef parameter of synchronization likelihood. Higher value leads to higher synchronization likelihood values.
    :param series_idx: Set series_idx to the desired index if used for the full ecog data (for SEEG individual file, leave as None)
    :param name: anything you would like to add to the saved filename
    :return: list of sync likelihood networks
    '''
    tseries, fs = get_tseries_and_fs(f, band = band, series_idx = series_idx)
    N, T = tseries.shape

    A_list = []
    for i in range(T // fs):
        A = np.zeros((N, N))
        t0 = i * fs
        tf = (i+1) * fs
        for j in range(N - 1):
            for k in range(j , N):
                A[j][k] = synchronizationLikelihood(tseries[j,t0:tf], tseries[k,t0:tf], pRef = pRef)
                A[k][j] = A[j][k]
                print(i, j, k, A[j][k])
        A_list.append(A)
    pk.dump(A_list, open("sync_likelihood_net_"+name+".pk", "wb")) #saves to pickle file.
    return A_list

def get_nat_freqs_from_tseries(tseries, Fs, look_frac = 1/3):
    '''
    :param tseries: (N,T) matrix
    :param look_frac: fraction of the timeseries to use for ngetting natural frequencies. Default 1/3
    :param Fs: sampling frequency
    :return: natural frequencies
    '''
    N, T = tseries.shape
    nat_freqs = np.array([freq_from_fft(tseries[i][:int(T * look_frac)], Fs) for i in range(N)])
    return nat_freqs

def plot_network(net):
    plt.figure()
    plt.axis('off')
    plt.rcParams.update({'font.size': 16})
    plt.title('t = 25 s')
    plt.rcParams.update({'font.size': 16})
    plt.imshow(net)
    plt.tight_layout()

if __name__ == '__main__':
    filepath_ecog = 'C:/Users/billy/PycharmProjects/BrainNetworks/Time_Evolving_Controllability_EC_Data/dataSets_clean.mat'
    fpath_131 = 'C:/Users/billy/PycharmProjects/BrainNetworks/Time_Evolving_Controllability_EC_Data/HUP131-short-ictal-block-1.mat'
    fpath_084 = 'C:/Users/billy/PycharmProjects/BrainNetworks/Time_Evolving_Controllability_EC_Data/HUP084-short-ictal-block-1.mat'

    f_ecog = h5py.File(filepath_ecog)
    f_131 = h5py.File(fpath_131)
    f_084 = h5py.File(fpath_084)

    A_list = construct_sync_likelihood_nets(0, f_ecog, series_idx=0)

    alpha, dt, totTime = 0.035, 1/512.0, 100

    tseries, fs = get_tseries_and_fs(f_ecog, band = None, series_idx=0)
    nat_freqs = get_nat_freqs_from_tseries(tseries, fs)

    syncs, maxes, means, times = get_timescale_max_data(A_list, nat_freqs,alpha, dt, int(totTime / dt))
    num_networks = len(A_list)
    aap.make_plots(num_networks, syncs, maxes, means, times, nat_freqs)


