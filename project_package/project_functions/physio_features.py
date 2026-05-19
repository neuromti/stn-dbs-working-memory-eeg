# -*- coding: utf-8 -*-
"""Physiological feature extraction utilities."""

import mne
import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import random
import pickle as pkl
from pathlib import Path

from .preprocessing import auto_identify_bads

#%% ################## TIME-FREQUENCY ANALYSIS ###############################


def tfa_epocher(data, t_window=2, dist_outliers=2.5):
    '''
    Epochs for the time-frequency analysis.

    Parameters
    ----------
    data : raw mne object
        Whole measurement, preprocessed.
    t_window : float, optional
        lenght of the time before and after the cue in seconds. The default is 2.
    dist_outliers : float, optional
        The times standard deviation allowed away from the median for the variance 
        of an epoch to be considereed outlier. The default is 2.5.

    Returns
    -------
    epochs : Epochs mne object
        Epochs of the ask-cue events in the measurement.

    '''
    cue = [cue for cue in data.annotations.description if 'ask' in cue][0]
    events, ev_descriptions = mne.events_from_annotations(data)
    epochs = mne.Epochs(data, events, event_id=ev_descriptions[cue], tmin=-t_window, 
                        tmax=t_window, baseline=(None, 0)).apply_baseline()
    #Automazized rejection of outliers
    epochs.drop(auto_identify_bads(epochs.get_data().mean(axis=1).std(axis=1), 
                                   dist=dist_outliers))
    return epochs

def correct_baseline(data, time, tmin, tmax):
    '''
    Baseline correction for spectrogram data.

    Parameters
    ----------
    data : numpy array, size(freqs, times)
        Third output of scipy.signal.spectrogram function.Array with the power
        values for frequency and time window.
    time : array
        Secound output of scipy.signal.spectrogram function. Array with the time
        series used for the spectrogram.
    tmin : float
        Start of baseline in seconds. The frame of closest time window of the 
        time array is used. Note that the 0 is in this case the actualy beginning
        of the epoch and not the cue. 
    tmax : float
        Start of baseline in seconds. Same condition as for tmin.

    Returns
    -------
    Numpy array
        Baseline corrected spectrogram data.

    '''
    
    fmin = np.argmin(np.abs(time - tmin))
    fmax = np.argmin(np.abs(time - tmax))
    # +1 so the last time bin meant is actually included
    baseline = data[:,fmin:fmax+1]
    # Adjust shape to make broadcast possible (replicate mean values over rows 
    # to match data's shape)
    baseline = np.array([np.full(data.shape[1], a) for a in baseline.mean(axis=1)]).reshape(data.shape)
    return np.subtract(data, baseline)



def correct_baseline_spectrogram(data, time, tmin, tmax):
    '''
    Baseline correction for spectrogram data.

    Parameters
    ----------
    data : numpy array, size(epochs, chans, freqs, times)
        Third output of scipy.signal.spectrogram function.Array with the power
        values for frequency and time window.
    time : array
        Secound output of scipy.signal.spectrogram function. Array with the time
        series used for the spectrogram.
    tmin : float
        Start of baseline in seconds. The frame of closest time window of the 
        time array is used. Note that the 0 is in this case the actualy beginning
        of the epoch and not the cue. 
    tmax : float
        Start of baseline in seconds. Same condition as for tmin.

    Returns
    -------
    Numpy array
        Baseline corrected spectrogram data.

    '''
    
    fmin = np.argmin(np.abs(time - tmin))
    fmax = np.argmin(np.abs(time - tmax))
    # +1 so the last time bin meant is actually included
    baseline = np.mean(data[:,:,:,fmin:fmax+1],-1,keepdims=True)
    baseline = np.tile(baseline, (1,1,1,data.shape[-1]))
    # Adjust shape to make broadcast possible (replicate mean values over rows 
    # to match data's shape)
    return data, baseline





def tf_data(epochs, freqBands=None, log_transform=True, baseline=(1,1.5)):
    '''
    Extract time-frequency data of epochs

    Parameters
    ----------
    epochs : Epochs mne object
        Epochs of the ask-cue events in the measurement.Output of tfa_epochs function
    freqBands : dict, optional
        Dictionary with name of frequency bands as keys and tuple with starting 
        and last frequency of the band. The default is None.
    log_transform : bool, optional
        To transfrom signal to dB. The default is True.
    baseline : tuple of two floats, optional
        Starting and finish pint of time window for besaline. See correct_baseline.
        The default is (1,1.5).

    Returns
    -------
    freqs : Numpy array
        frequency bins from the spectrogram (1-30Hz).
    time : Numpy array
        time bins from the spectrogram.
    tfdata : Numpy array
        spectrogram matrix.
    tfband : dict with Numpy arrays
        the keys are the frequency band names, the values are 1-D arrays with     
        time frequency data by frequency band, only if freqBands is given.
    tferror : Numpy array
        the keys are the frequency band names, the values are 1-D arrays with     
        time frequency data error by frequency band, only if freqBands is given.
    '''
    sfreq = epochs.info['sfreq']
        
    freqs, time, Sxx = sg.spectrogram(epochs.get_data(),
                                      nperseg=int(sfreq), nfft=sfreq, fs=sfreq,
                                      noverlap=int(sfreq*0.95))
    # Transform, extrcat domain of interest and correct
    if log_transform:
        Sxx = 10*np.log10(Sxx)
        
    #subtract baseline; only keep frequencies 1 to 30 Hz
    Sxx, BL = correct_baseline_spectrogram(Sxx[:,:,1:31,:], time, baseline[0], baseline[1]) #dimensions: epochs, channels, frequencies, time

    Sxx_BL = np.subtract(Sxx,BL)
    #average over epochs
    tfdata = Sxx_BL.mean(axis=0) #format channels x freq x time
    #TODO: averaging functions must be adapted
    
    # Store power by frequency band, optional
    
    if any(freqBands):
        tfband = {}
        tferror = {}
        for b in freqBands:
            data = tfdata[:,freqBands[b][0]:freqBands[b][1], np.floor(len(time)/2).astype(int):]#select second half of epoch (from query stim on) for freq band of interest
            tfband[b] = np.median(data) #average over freqs, time, and chans
            #tferror[b] = data.std(axis=1)/np.sqrt(data.shape[1])
        
        return freqs[1:31], time, Sxx, BL, tfband
    else:
        return freqs[1:31], time, tfdata
    
    

def welch_data(epochs, freqBands=None, log_transform=True):
    '''
    Extract time-frequency data of epochs

    Parameters
    ----------
    epochs : Epochs mne object
        Epochs of the ask-cue events in the measurement.Output of tfa_epochs function
    freqBands : dict, optional
        Dictionary with name of frequency bands as keys and tuple with starting 
        and last frequency of the band. The default is None.
    log_transform : bool, optional
        To transfrom signal to dB. The default is True.
    baseline : tuple of two floats, optional
        Starting and finish pint of time window for besaline. See correct_baseline.
        The default is (1,1.5).

    Returns
    -------
    freqs : Numpy array
        frequency bins from the spectrogram (1-30Hz).
    time : Numpy array
        time bins from the spectrogram.
    tfdata : Numpy array
        spectrogram matrix.
    tfband : dict with Numpy arrays
        the keys are the frequency band names, the values are 1-D arrays with     
        time frequency data by frequency band, only if freqBands is given.
    tferror : Numpy array
        the keys are the frequency band names, the values are 1-D arrays with     
        time frequency data error by frequency band, only if freqBands is given.
    '''
    sfreq = epochs.info['sfreq']
        
    freqs, Sxx = sg.welch(epochs.get_data(),axis=-1,
                                      nperseg=int(sfreq), fs=sfreq,
                                      noverlap=int(sfreq*0.95))
    # Transform, extrcat domain of interest and correct
    if log_transform:
        Sxx = 10*np.log10(Sxx)
    return freqs, Sxx


def visualize_tf(tfband, tferror, time, t_corr=0.2, bands='all'):
    '''
    

    Parameters
    ----------
   tfband : Numpy array
        time frequency data by frequency band, only if freqBands is given.
        Output tf_data function
    tferror : Numpy array
        time frequency data error by frequency band, only if freqBands is given.
        Ouput tf_data function
    time : Numpy array
        time bins from the frequency data
    t_corr : float, optional
        time correction on the time (original time in epoch before stimulus event). 
        The default is 0.2
    bands : list of str, optional
        list with bands to be visualized, they have to be keys in tfband. 
        The default is 'all', in which case all bands (+total) are plotted

    Returns
    -------
    None.

    '''
    if bands != 'all':
        tfband = {k:v for k,v in tfband.items() if k in bands}
        tferror = {k:v for k,v in tferror.items() if k in bands}
    
    plt.figure()
    
    color = ['deepskyblue','forestgreen' ,'steelblue', 'indigo', 'mediumblue']
    i=0
    for b in tfband:
        plt.plot(time-t_corr, tfband[b], label=str(b), c=color[i])
        plt.fill_between(time-t_corr, tfband[b]-tferror[b], tfband[b]+tferror[b], alpha=0.3,
                         color=color[i])
        i += 1
        
    plt.ylabel('power [dB]')
    plt.xlabel('time [s]')
    plt.legend()
    
def meanpower(datadict, time=None, timewindow=None):
    '''
    Extract the mean power by frequency band for certain time window.

    Parameters
    ----------
    datadict : dict
        it is the same tfband dictionary outputted by tf_data.
    time : Numpy Array, optional
        time bins from time frequency data, output time from tf_data. 
        The default is None.
    timewindow : tuple of floats, optional
        tuple of length two with initial and end time (parting from start of
        the frequency data). as the time frquency data has fixed frequency bins
        it takes the nearest ones. The default is None.

    Returns
    -------
    meanDict : dict
        dictionary with frequency bands names as keys and the mean power value 
        computed as value.

    '''
    if any(timewindow):
        lims = (np.argmin(np.abs(time - timewindow[0])), 
                np.argmin(np.abs(time - timewindow[1])))
        datadict = {k:v[lims[0]: lims[1]] for k,v in datadict.items()}
    meanDict = {k:v.mean() for k,v in datadict.items()}
    return meanDict

########################## P3 COMPONENT ANALYSIS #############################
# Manually chosen cluster information, extracted for further analysis
# Running the permutations cluster test lasts very long, this is why the cluster 
# numbers and electrodes are hardcoded here. 
clustInfo = {'acoustic': {'cluster': 46, 'chans': ['TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 
                                              'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 
                                              'Oz', 'O2', 'PO10', 'AF7', 'AF3', 'AF4', 
                                              'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 
                                              'FT7', 'FC3', 'FC4', 'FT8', 'FT10']},
             'visual': {'cluster': 69, 'chans': ['T8', 'TP9', 'CP5', 'CP1', 'CP2', 
                                              'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 
                                              'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 
                                              'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 
                                              'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6']}}  


def save_cluster_info(path):
    """Write manually selected P300 cluster information to disk."""
    with Path(path).open('wb') as file:
        pkl.dump(clustInfo, file)


def rearrange_events(epochs, oddball):
    '''
    Rename oddball events identifiers from measurements to make it possible to 
    concatenate epochs

    Parameters
    ----------
    epochs : mne Epochs object
        preprocessed epochs. 
    oddball : 'acoustic' or 'visual'
        measurement type ran along with the oddball measurement being analyzed.

    Returns
    -------
    epochs : mne Epochs object
        same object with new ids for the oddball events.

    '''
    matchDict = {'acoustic':{'distractor': 1, 'lowPitch': 2, 'highPitch': 3 },
                 'visual':{'distractor': 1, 'circle': 2, 'large': 3, 'largeCircle': 3 }}
    matchDict = matchDict[oddball]
    oldDict = epochs.event_id
    events = epochs.events
    rematch = {num:matchDict[old] for old,num in oldDict.items()}
    print(rematch)
    nId = np.array([rematch[e] for e in epochs.events[:,2]])
    events[:,2] = nId
    epochs.events = events
    epochs.event_id = matchDict
    return epochs



def epochs_grouper(files, oddball):
    '''
    Groups epochs of several measurements by stimulus type

    Parameters
    ----------
    files : list of str
        List with the paths to the preprocessed .fif EEG files. Main file!
        This means the ones with ending _p_raw.fif as outputted by the 
        get_available_measurements function
    oddball : 'acoustic' or 'visual'
        type of oddball being analyzed.
    
    Returns
    -------
    allepochs : TYPE
        DESCRIPTION.

    '''
    # To later rename oddball by their type
    oddCategory = {'circle':'nontarget',
               'lowPitch': 'nontarget',
               'large': 'target',
               'largeCircle': 'target',
               'highPitch': 'target',
               'distractor': 'distractor'}

    allepochs = {}
    for i,file in enumerate(files):
        efile = file.replace('_p_raw', '_oe_epo')
        epochs = mne.read_epochs(efile, preload=True).apply_baseline()
        #Rename oddball type identifiers to make it possible to concatenate epochs
        epochs = rearrange_events(epochs, oddball)
        for k in epochs.event_id.keys():
            name = oddCategory[k]
            if name == 'nontarget':
                #Subsampling randomly as nontarget epochs are too much 
                n_ep = epochs[k].__len__()
                epochs[k].drop(random.sample(range(n_ep), n_ep-20))
            #Store in dictionary by oddball type
            if i == 0:
                allepochs[name] = epochs[k]
            else:
                allepochs[name] = mne.concatenate_epochs([allepochs[name], epochs[k]])
    return allepochs

def epochs_permutation_test(fileList, oddball, clustInfo=False):
    '''
    Performs cluster permutation test and if a preselected cluster is given 
    it plots the information for it
    

    Parameters
    ----------
    fileList : list of str
        list with paths.
    oddball : 'acoustic' or 'visual'
        oddball type analyzed
    clustInfo : dict, optional
        dict with key with name of oddball and subkey 'cluster' with number of
        cluster selected. The default is False, in which case nothing is plotted.

    Returns
    -------
    permutations test output. Check mne.cluster_permutation_test
    F_obs : 
        DESCRIPTION.
    clust : TYPE
        DESCRIPTION.
    cluster_pv : TYPE
        DESCRIPTION.
    H0 : TYPE
        DESCRIPTION.

    '''

    # Choose only epochs of V0, because concatented epochs cannot have different PCA
    # projections (SSP filter)
    files = [f for f in fileList if 'V0' in f]
    allepochs = epochs_grouper(files, oddball)
    
    #TODO: make adjacencys tructure!!
    # Axes swapped twice to get the convention  of permutation test          
    adj, ch_names = mne.channels.find_ch_adjacency(allepochs['distractor'].info,None)
    epochs_1 = np.swapaxes(allepochs['distractor'].get_data(),1,2)
    epochs_2 = np.swapaxes(allepochs['target'].get_data(),1,2)
    F_obs, clust, cluster_pv, H0 = \
        mne.stats.permutation_cluster_test([epochs_1,epochs_2], n_permutations=1000, adjacency = adj)
        
       
    #Visualization  
    
    # Previously chosen significant cluster, electrodes that belong to this cluster
    #within the time of interest are selected
    if clustInfo != False:
        cluster = clust[clustInfo[oddball]['cluster']] # SDR: 69, VD: 46
        time = np.array([0.45,0.65]) # time window of interest
        sfreq = allepochs['distractor'].info['sfreq']
        time = np.array(sfreq*time).astype(int)
        c_chans = [n for i,n in enumerate(allepochs['distractor'].ch_names) if any(cluster[time[0]:time[1],i])]
        print(c_chans)
        for k in allepochs.keys():
            temp_epoch = allepochs[k].pick_channels(c_chans)
            temp_epoch.average().plot_joint(times=[0.0, 0.32, 0.4, 0.5],
                                     title= oddball + k +', cluster electrodes',
                                     ts_args={'gfp':True})
    return F_obs, clust, cluster_pv, H0

def get_P300(epochs, start=0.2, lims=(0.45, 0.65), pick_channels=False):
    '''
    Given epochs it identifies the biggest peak in the selected interval and 
    returns its features and important information for the analysis or 
    visualization. It extracts the P3 wave from th global field potential of
    the picked channels.

    Parameters
    ----------
    epochs : mne Epochs object
        preprocessed epochs linked to oddball stimulus.
    start : float, optional
        Baseline duration in seconds. The default is 0.2.
    lims : tuple of floats, optional
        The limits of the time window to extract the P3 from in s. Assuming zero at 
        the begin of the epoch, not stimulus displayed! The default is (0.45, 0.65),
        which with the default baseline of 0.2, means 250ms to 450ms after stimulus.
    pick_channels : list of str, optional
        Subset of channels to pick. The default is False, in which case all 
        channels are used.

    Returns
    -------
    peak : int
        index of identified peak in array.
    latency : float
        peaking time of P3 after stimulus in seconds.
    amplitude : float
        Amplitude of P3, value of array at point given by peak.
   

    '''
    if pick_channels != False:
        epochs.pick_channels(pick_channels)  
    sfreq = epochs.info['sfreq']

    #Compute GFP of given channels and get the peaks
    evoked = epochs.apply_baseline().get_data().mean(axis=0).std(axis=0)
    peaks_p, _ = sg.find_peaks(evoked)

    peaks = np.array([p for p in peaks_p if lims[0]*sfreq < p < lims[1]*sfreq])
    if len(peaks) == 0: 
        #this happened only once, so this allows to pick the next peak after the
        #set window
        peaks = [np.min(peaks_p[peaks_p > lims[1]*sfreq])]
    #Note: argmax returns a mask
    peak = peaks[np.argmax(evoked[peaks])]
    
    amplitude = evoked[peak]
    latency = peak/sfreq - start
    return peak, latency, amplitude


def visualize_P300(epochs, peak, start=0.2, pick_channels=False,
                   error=True):
    '''
    Visualize P3 potentials with the selcted peak marked

    Parameters
    ----------
    epochs : mne Epochs object
        preprocessed epochs linked to oddball stimulus.
    peak : int
        index of identified peak in array. Output of get_P300
    start : float, optional
        Baseline duration in seconds. The default is 0.2.
    pick_channels : list of str, optional
        Subset of channels to pick. The default is False, in which case all 
        channels are used.

    Returns
    -------
    None.

    '''
    if pick_channels != False:
        epochs.pick_channels(pick_channels)  
    evoked = epochs.get_data().mean(axis=0).std(axis=0)
    sfreq = epochs.info['sfreq']
    plt.figure()

    
    plt.plot(np.linspace(-0.2, 1, 6001), evoked, color='seagreen', label='GFP over electrodes')
    plt.plot(peak/sfreq-start, evoked[peak], 'rx', label='Peak maximum')
    plt.xlabel('time [s]')
    plt.ylabel('Current source density [mV/m^2]')
    plt.legend()
    
    
