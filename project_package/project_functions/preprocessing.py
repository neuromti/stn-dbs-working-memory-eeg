# -*- coding: utf-8 -*-
"""Preprocessing utilities for the working-memory EEG analysis."""

from pathlib import Path

import mne
import pandas as pd
import numpy as np
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
import pickle as pkl 



#%%

####################  LOADING INFORMATION AND DATA ###########################

def load_patient_meas(path_to_folder, file_name):
    '''
    Load the eeg data in "brainvision" format from the source file

    Parameters
    ----------
    path_to_folder : str
        Path to the folder where all data has been saved as .fif files.
    file_name : str
        Name of the file.

    Returns
    -------
    data: mne Raw object
    data_path : TYPE
        DESCRIPTION.

    '''
    data_path = Path(path_to_folder) / file_name
    print('Accessing', data_path)
    return  mne.io.read_raw_fif(data_path, preload=True), str(data_path)


def load_test_patient_meas_pkl(data_path):
    '''
    Loads test events from pkl file 

    Parameters
    ----------
    data_path : str
        Of the vhdr file of the corresponding measurement. (See load_patient_meas)

    Returns
    -------
    test: Double list
        Unpickled double list of events and stimulus of the test.

    '''
    test_path = Path(str(data_path).replace('.vhdr', '.pkl'))
    with test_path.open("rb") as file:
        return pkl.load(file)


def load_test_patient_meas(data_path):
    ''' 
    Loads test events from csv file
    
    data_path : str
        Of the vhdr file of the corresponding measurement. (See load_patient_meas)

    Returns
    -------
    test_info: pandas DataFrame
        Unpickled double list of events and stimulus of the test. '''
    test_path = str(data_path).replace('raw.fif', '_test.csv')
    return pd.read_csv(test_path, index_col=0)


def get_stim_info(data_path, trial_name):
    '''
    

    Parameters
    ----------
    data_path : str
        Data path of .csv file containing stimulation info.
    trial_name : str
        Name of the trial in format: (patient ID)_V(measurement)_(setting)
        Setting is just for measurements 3
        Example: 3_V4 or 5_V3_2

    Returns
    -------
    stimulation: str
        string with the stimulation type of this measurement

    '''
    
    info = pd.read_csv(data_path, index_col=0)
    cut = trial_name.find('_')
    patient, col = trial_name[:cut], trial_name[cut+1:]
    print('Stimulation state in this measurement is: ', info.loc[int(patient)][col])
    return info.loc[int(patient)][col]
    
    
########################## VISUALIZATIONS ####################################

def visualize_filtered_channel(data, filt_data, channel, Fs, patient,
                               measurement, removed_peaks=np.array(False),
                               xlim=500):
    '''
     Generates two plots with the original and the filtered signal of one 
     channel: in frequency domain and in time domain.
    Note that for this function you should have performed the filtering on a 
    copy of the original object

    Parameters
    ----------
    data : mne raw object 
        The ORIGINAL DATA before being filtered.
    filt_data : mne raw object
        The filtered data.
    channel : int
        The number of the channel to be visualized.
    Fs : int
        Sampling frequency.
    patient : int
        Patient's ID.
    measurement : int
        Measurement's ID.
    removed_peaks : array, optional
        Array with the frequencies at which notch filters were applied. They 
        will be then also ploted as 'x' in the psd plot. The default is 
        np.array(False).
    xlim : number, optional
        Limit at the x-axis. The default is 500.

    Returns
    -------
    None.

    '''
    
    #Frequency domain
    plt.figure(0)
    plt.psd(data[channel], NFFT=Fs, Fs=Fs, label='original signal')
    plt.psd(filt_data[channel], NFFT=Fs, Fs=Fs, label='filtered signal')
    if removed_peaks.any() :
        _,power = scipy.signal.welch(data[channel], nperseg = Fs, fs = Fs, 
                                      nfft = Fs)
        plt.plot(removed_peaks, 10*np.log10(power[removed_peaks]),
                  'x', label='removed peaks')
    plt.legend()
    plt.title('Patient:' + str(patient) + ' Measurement:' + str(measurement) +
              '  Channel:' + str(channel))
    plt.xlim([0,xlim])
    
    #Time domain
    plt.figure(1)
    t = np.arange(0,len(data[channel])/Fs, 1/Fs)
    plt.plot(t, data[channel], label='Original signal')
    plt.plot(t, filt_data[channel], label='Filtered')
    plt.title('Patient:' + str(patient) + ' Measurement:' + str(measurement) +
              '  Channel:' + str(channel))
    plt.xlabel('Time in s')
    plt.legend()
    if xlim == None: xlim = max(t)
    plt.xlim([0, xlim/Fs])


################### DETECTION OF BAD CHANNELS/EPOCHS #########################

def pick_bads(variances, names, median, std, max_var, min_var, idx, dist=3,
              plot_timeout=20, title='Pick bads'):
    '''
    Manual selection of bad elements based on scatter plot with reference of 
    elements to the median a standard deviations

    Parameters
    ----------
    variances : array
        Standard deviations of the elements of a single series to be evaluated.
    names : list of str
        Names of the elements for them to be labeled in the plot.
    plot_timeout : number, optional
        Time in second that the programm will stop running during the element
        manual picking in the plot. The default is 20.
    title : str, optional
        Ttile that the plot should show. The default is 'Pick bads'.

    Returns
    -------
    events : list of int
        List with the indexes of the elements selected.

    '''
    plt.ion()
    num_elems = np.arange(len(variances)) 
    
    events = []
    
    def onpick(event):
        ind = event.ind
        points = tuple(zip(num_elems[ind], variances[ind]))
        print('Onpick points:', points)
        print('Name/type of point selected:', names[ind[0]])
        if len(ind):
            events.append(int(ind[0]))
            print('Events selected for now:', events)
        return events
    
    def dispose(event):
        fig.canvas.stop_event_loop()
        print ("disposed")
    
    
    fig = plt.figure()
        
    plt.scatter(num_elems, variances, color=['g' if i else 'r' for i in idx], 
                      picker=True)
    plt.axhline(y=median)
    plt.axhline(y=max_var, color='r')
    plt.axhline(y=min_var, color='r')
    plt.axhline(y=median+2*std, color='lightgrey')
    plt.axhline(y=median-2*std, color='lightgrey')
    plt.axhline(y=median+std, color='lightgrey')
    plt.axhline(y=median-std, color='lightgrey')
    plt.title(title)    
    
    for i, txt in enumerate(names):
        plt.annotate(txt, (num_elems[i], variances[i]))
        
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.canvas.start_event_loop(timeout=plot_timeout)
    fig.canvas.mpl_connect('close_event', dispose)
    #plt.show(block=True)

    return events

def auto_identify_bads(array, dist=3):
    '''
    Outlier identification without manual intervention
    Checks if there are more then three elements in the input array before operating

    Parameters
    ----------
    array : Numpy array
        Array with values to be tested on outliers.
    dist : number, optional
        times standard deviations away from median to be accepted. The default is 3.

    Returns
    -------
    outliers: Numpy array
        Array with the identified outliers.

    '''
    median = np.median(array)
    if array.size > 3:
        std = np.std(array)
        #times of std away of mean allowed. Definition of outliers suggests 3
        max_var = median + dist * std
        min_var = median - dist * std
        
        idx = (array < max_var) & (array > min_var)
        return np.arange(len(array))[~idx].astype(int)
    else:
        return np.array([])


def identify_bads(variances, dist=3, manual=True, names=None, plot_timeout=20, 
                  title='Pick bads'):
    '''
    Automated and manual identification of outliers. Automated part is base on 
    distance to the median, manual part on manual event picking in scatter plot.

    Parameters
    ----------
    variances : array 
        Array with elements from which to identify the outliers.
    names : list of str
        Names of the single elements.
    plot_timeout :number, optional
        Time in second that the programm will stop running during the element
        manual picking in the plot. The default is 20.
    title : str, optional
        Title that the plot should show. The default is 'Pick bads'.

    Returns
    -------
    bad_idx : array
        Index of the elements selected as outliers.

    '''
    # Establish interval of accepted variance
    median = np.median(variances)
    std = np.std(variances)
    #dist: times of std away of mean allowed. Definition of outliers suggests 3
    max_var = median + dist * std
    min_var = median - dist * std
    
    idx = (variances < max_var) & (variances > min_var)
    num_elems = np.arange(len(variances)) 
    
    if manual:
        man_events = pick_bads(variances, names,median, std, max_var, min_var, idx,
                                plot_timeout=20, title=title)
        bad_idx = np.append(man_events, num_elems[~idx])
    else:
        bad_idx = num_elems[~idx]
        
    print('Rejected elements:', bad_idx)
    return bad_idx.astype(int)


def get_bad_oddball_epochs(epochs, dist=3, manual=True):
    '''
    Gives index of epochs considered as outliers due to variance within the 
    epoch. It is optimized to do the outlier search grouping the epochs by stimulus.

    Parameters
    ----------
    epochs : mne Epochs object
        oddball Epochs extracted after main preprocessing.
    dist : number, optional
        times standard deviations away from median to be accepted. The default is 3.
    manual : bool, optional
        whether manual identification should be also performed. The default is True.

    Returns
    -------
    outliersIdx: Numpy Array
        Indexes of the epochs classified as outliers.

    '''
    all_idx = []
    
    # Ensure identification in the same condition (stimulus)
    for key in epochs.event_id.keys():
        # Index within the whole oddball set of the epochs corresponding to the
        # current key
        key_idx = [i for i in range(epochs.__len__()) if 
                   list(epochs[i].event_id.keys())[0] == key]
        # Compute variance vlues for each epoch (std)
        var = np.std(np.mean(epochs[key].get_data(), axis=1), axis=1)
        key_bads = identify_bads(var, dist=dist, manual=manual, names=key_idx, 
                                 title=str(key))
        #Translate from idx within the stimulus group group to overall index
        all_idx.extend([key_idx[j] for j in key_bads])
    print('Epochs chosen as bad: ', all_idx)
    # Return indexes as integers to facilitate ctual use as index
    return np.array(all_idx).astype(int)

######################### MARKERS & EPOCHS HANDLING ###########################

def rename_markers (annotations, start_point=0):
    '''
    Extracts the relevant event name from the annotations.
    Explanation: the events IDs from the recordings were saved as 'Showing [stimulus]
    at time [time in s]'. The goal of the function is to only get the stimmulus name
    to group epochs related to the same stimulus
    
    Parameters
    ----------
    annotations : mne Annotations object
        Original annotations saved in the Raw data.
    start_point : int, optional
        Index from which to start the change auf the names. The default is 0.

    Returns
    -------
    event_names : list of str
        The new descriptions of the annotations containing solely relevant 
        information.

    '''
    event_names = annotations.description[start_point::]

    for idx, name in enumerate(event_names):
        if not name.startswith('Stimulus'):
            name = name[name.find("Showing")::].split()[1]
            event_names[idx] = name
    return event_names



def epoch_duration(annotations):
    '''
    Computes the duration of the epochs related to the tasks VDR and SDR

    Parameters
    ----------
    annotations : mne Annotations object
        Annotations in the data (already transformed by rename_markers).

    Returns
    -------
    epochs_duration : number
        Epoch duration in seconds.

    '''
    cue_time = []
    ask_time = []
    for idx, name in enumerate(annotations.description):
        if name.startswith('cue'): cue_time.append(idx)
        if name.startswith('ask'): ask_time.append(idx)  
    # Compute epoch duration beginning with meorizing cue and ending with 
    # query cue
    onset_time = np.array(annotations.onset)[cue_time]
    end_time = np.array(annotations.onset)[ask_time]
    epochs_dur = end_time - onset_time
    print('There are ', len(epochs_dur), ' in this trial')
    return epochs_dur.mean()



def oddball_names (events_dictionary):
    '''
    Returns a subdictionary that includes only the markers related to oddballs

    Parameters
    ----------
    events_dictionary :  dict
        Dictionary with the events descriptions and their assigned ID. Normally
        the second elemented of the tuple outputted by mne.events_from_annotations()
    Returns
    -------
    oddball_dictionary: dict
        Dictionary with oddball names a keys.

    '''
    oddball_names = ['circle', 'distractor', 'large','highPitch','lowPitch',
                     'largeCircle', ]
    return {k:v for k,v in events_dictionary.items() if k in oddball_names}

def multi_find(string, substring):
    '''
    Adaptation of find function for string but for all time the substring is present
    in the string and not only the first one.
    
    It is useful in the handling of automated filenames

    Parameters
    ----------
    string : str
        big string where the substring is supposed to be.
    substring : str
        substring to be look for in the string.

    Returns
    -------
    substringIdx: list of int
        (starting) indexes of the substring inside the string.

    '''
    return [i for i in range(len(string)) if string.startswith(substring, i)]

def epoching(path, p_dir, oddball=True, task=True):
    '''
    Performs only the epoching part of the preprocessing. It extracts the epochs, 
    rejects the outliers and saves the mne Epochs objects. It returns a dictionary with 
    the indexes of the dropped epochs according to epoch type.
    
    It is useful when only this part needed to be repeated over several measurements.

    Parameters
    ----------
    path : str
        path of the main preprocessed file (according to preprocessing convention,
        those which ended with _p_raw.fif).
    p_dir : str
        path to the directory where the obtained epochs files should be saved to.
    oddball : bool, optional
        whether oddball epochs whould be extracted. The default is True.
    task : bool, optional
        whether main task (delayed response task) epochs are to be extracted. 
        The epochs obtained here start with the display cue and end with que query cue.
        The default is True.

    Returns
    -------
    droppedEpochs : dict
        indexes of rejected outlier epochs.

    '''
    path = Path(path)
    p_dir = Path(p_dir)
    task = 'VDR' if 'VDR' in str(path) else 'SDR'
    data = mne.io.read_raw_fif(path)
    events, ev_descriptions = mne.events_from_annotations(data)
    
    save_path = p_dir / path.name.replace('_raw.fif', '')
    droppedEpochs = {}
    
    if task:    
        start_cue = [cue for cue in ev_descriptions.keys() if 'cue' in cue]
        duration = epoch_duration(data.annotations)
        task_epochs = mne.Epochs(data, events, event_id=ev_descriptions[start_cue[0]], 
                                     tmin=0, tmax=duration+1, baseline=None)
        
        oddball_epochs = mne.Epochs(data, events, event_id=oddball_names(ev_descriptions), 
                                      tmin=-0.2, tmax=+1, baseline=None)
        
        #Filter bad epochs
        var_1 = np.std(np.mean(task_epochs.get_data(), axis=1), axis=1)
        bad_task_epochs = identify_bads(var_1, task_epochs.events[:,2],
                                    title=task + str(task_epochs.event_id))
        task_epochs.drop(bad_task_epochs).save(f'{save_path}_te_epo.fif',
                                                     overwrite=True)
        droppedEpochs['mainTask'] = bad_task_epochs
        
    if oddball:
        bad_oddball_epochs = get_bad_oddball_epochs(oddball_epochs.drop_bad())
        oddball_epochs.drop(bad_oddball_epochs).save(f'{save_path}_oe_epo.fif',
                                                     overwrite=True )
        droppedEpochs['oddball'] = bad_oddball_epochs
        
    plt.close('all')
    
    return droppedEpochs

################### TIMING CORRECTION VDR ####################################

def correct_VDR_timing(annotations):
    '''
    Time information correction of the auditory oddball.
    
    Explanation: pitches of the auditory oddball were by error registered one 
    second after being played. This function fixes this to have reliable markers
    that can be used in the behavioral analysis

    Parameters
    ----------
    annotations : mne Annotations object

    Returns
    -------
    newVec : Numpy array
        Array containing the corrected tming of all events in annotations.

    '''
    #auditory oddball names
    aobNames = ['distractor', 'highPitch','lowPitch']
    #Correct time of oddball related annotations, keep original times of 
    #everything else
    newVec = np.array([a['onset'] - 1 if a['description'] in aobNames else a['onset'] for a in annotations])
    return newVec
        
        
#################### PREPROCESSING COMPLETE ##################################

def preprocess(data_folder, data_file, p_data_folder, stim_info_file, manual=True,
               bad_ch = None, plot_filtered=False):
    '''
    Applies the whole preprocessing pipeline to trials' data. In the returned 
    dictionary there is info about data path and rejected channels and epochs

    Parameters
    ----------
    data_folder : str
        Path to folder where the unprocessed data is in.
    data_file : str
        File name of the data to be preprocessed.
    p_data_folder : str
        Path to falder where prerpocessed that should be stored.
    manual : bool
        whether outlier of epochs and channels should (in addition to automatized
        identification) be also selected manually base on scatter plots
    bad_ch : list or None
        list with bad channels, if None it runs the bad channel identification
    plot_filtered : bool, optional
        If True, the final resault of the preprocessed complete measurement is 
        plotted. The default is False.

    Returns
    -------
    prep_info : dict
        Dictionary with information about outcomes of the proprcessing, as well
        as data path.

    '''    
    
    p_data_folder = Path(p_data_folder)
    p_data_folder.mkdir(parents=True, exist_ok=True)

    #EEG & trial info
    data, data_path = load_patient_meas(data_folder, data_file)
    stimulation = get_stim_info(stim_info_file, data_file[:-11])
    Fs = int(data.info['sfreq'])
    task = 'VDR' if 'VDR' in data_file else 'SDR' 
    
    #Drop EMG channels, unnecessary channels (product of double indexing) and time
    #channel, which has false information
    remove_names = np.array(['LB', 'LB1', 'LB2', 'LT', 'RB', 'RB1', 'RB2',
                  'RT', 'Gonio'])
    remove_names = remove_names[np.isin(remove_names, data.ch_names)]
    data = data.drop_channels(list(remove_names))
    
    #Set montage & reference 
    data.set_montage('standard_1005')
    
    #Save current annoration for later
    markers = data.annotations
    
    #Compute SSP projections in their "most original" state
    #if stimulation != 'OFF':
    dbs_peaks, dbs_properties = scipy.signal.find_peaks(data.get_data()[0], distance=35,
                                                        width=1)
    data.set_annotations(mne.Annotations(dbs_peaks/Fs, 1/Fs, 'DBS'))
    dbs_epochs = mne.Epochs(data, mne.events_from_annotations(data)[0], tmin=-18/Fs, 
                            tmax=18/Fs) #half of the epoch
    dbs_epochs.average()#.plot()
    
    #Noise frequencies
    pln_freqs = np.arange(50, 251, 50)
    dbs_freqs = np.array([128, 255, 383])
    
    # Apply filters
    data.notch_filter(pln_freqs).notch_filter(dbs_freqs,trans_bandwidth=6).filter(0.3,35)#
    
    # Automated and manual bad channels identification
    variance = np.std(data.get_data(), axis=1)
    
    #Change for automation
    if bad_ch != None:
        bad_ch = bad_ch
    else:
        bad_ch_idx = identify_bads(variance, dist=2, manual=manual, names=data.ch_names, plot_timeout=14)
        bad_ch = [data.ch_names[i] for i in bad_ch_idx]
        
    # Interpolate bad channels
    data.info['bads'].extend(list(bad_ch))
    print('Channels to interpolate:', bad_ch)
    data.interpolate_bads()
    
    #CSD
    mne.preprocessing.compute_current_source_density(data, stiffness=3, copy=False)
    
    # #Projections
    # if stimulation != 'OFF':
    #     data.add_proj(mne.compute_proj_epochs(dbs_epochs)).apply_proj()
    
    #Manage annotations
    data.annotations.delete(range(len(data.annotations)))
    data.set_annotations(markers) 
    data.annotations.description[1::] = rename_markers(data.annotations, 
                                                       start_point=1)
    
    # Correct timing of the acoustic oddball (1s)
    if task == 'VDR':
        data.annotations.onset = correct_VDR_timing(data.annotations)
    
    #Save
    save_path = p_data_folder / data_file.replace('raw.fif', '')
    data.save(f'{save_path}_p_raw.fif', overwrite=True )
    
    #Epoching according to analysis
    events, ev_descriptions = mne.events_from_annotations(data)
    
    start_cue = [cue for cue in ev_descriptions.keys() if 'cue' in cue]
    duration = epoch_duration(data.annotations)
    task_epochs = mne.Epochs(data, events, event_id=ev_descriptions[start_cue[0]], 
                                 tmin=0, tmax=duration+1, baseline=None)
    
    oddball_epochs = mne.Epochs(data, events, event_id=oddball_names(ev_descriptions), 
                                  tmin=-0.2, tmax=+1, baseline=None)
    #identify_bads(variances, dist=3, manual=True, names=None, plot_timeout=20, 
    #              title='Pick bads')
    #Filter bad epochs
    var_1 = np.std(np.mean(task_epochs.get_data(), axis=1), axis=1)
    bad_task_epochs = identify_bads(var_1, dist=2.5, manual=manual, 
                                    names=task_epochs.events[:,2],
                                    title=task + str(task_epochs.event_id))
    
    bad_oddball_epochs = get_bad_oddball_epochs(oddball_epochs.drop_bad(), dist = 2.5, 
                                                manual=manual)
    
    #And save :)
    oddball_epochs.drop(bad_oddball_epochs).save(f'{save_path}_oe_epo.fif',
                                                 overwrite=True )
    task_epochs.drop(bad_task_epochs).save(f'{save_path}_te_epo.fif',
                                                 overwrite=True)
    #And plot?
    if plot_filtered: 
        data.plot()
    
    #Create dictionary to facilitate storing through iterations in data frame
    prep_info = {'measurement':data_file.replace('raw.fif', ''), 
                 'task': task, 
                 'n_bad_ch': len(bad_ch),
                 'bad_ch': bad_ch,
                 'n_task_epochs': task_epochs.__len__(),
                 'n_bad_task_epochs': len(bad_task_epochs),
                 'n_oddball_epochs': oddball_epochs.__len__(),
                 'n_bad_oddball_epochs': len(bad_oddball_epochs),
                 'path': str(save_path)}
    
    plt.close('all')
    return prep_info
