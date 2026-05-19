# -*- coding: utf-8 -*-
"""End-to-end workflow for the working-memory EEG analysis."""

#%%
import ast
import seaborn as sns
from pathlib import Path
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mne
from scipy import stats
from project_package.project_functions.preprocessing import (preprocess,
                                                             get_stim_info)
from project_package.project_functions.behavioral_features import (evaluate_beh_patients,
                                                                   summarize_all_results)
from project_package.project_functions.files_management import (get_available_measurements,
                                                                divide_by_measurement_type)
from project_package.project_functions.physio_features import (get_P300, clustInfo,
                                                               epochs_permutation_test,
                                                               tfa_epocher, meanpower,
                                                               tf_data)
from project_package.project_functions.statistics import (add_norm_values,
                                                          add_baseline_values,
                                                          corr_df, subsetter,
                                                          significance_df, renamer,
                                                          plot_for_stats)
plt.rcParams["font.family"] = "Arial"


#%% Define steps to be ran

redo_preprocessing = False
redo_timefrequency = False
redo_permutation_test = False
redo_P300 = False
redo_behavioral = False
plots_report = False
plotP3waves = False

#%% PATHS
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
METADATA_DIR = DATA_DIR / 'metadata'
RAW_DATA_DIR = DATA_DIR / 'raw'
PREPROCESSED_DATA_DIR = DATA_DIR / 'preprocessed'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
R_INPUT_DIR = DATA_DIR / 'r_inputs'
RESULTS_DIR = PROJECT_ROOT / 'results'
STATISTICS_RESULTS_DIR = RESULTS_DIR / 'statistics'
FIGURES_DIR = RESULTS_DIR / 'figures'

for folder in (PREPROCESSED_DATA_DIR, PROCESSED_DATA_DIR, R_INPUT_DIR,
               STATISTICS_RESULTS_DIR, FIGURES_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# Measurement and task metadata.
subsetInfoPath = METADATA_DIR / 'Analizable_info.csv'
stimInfoPath = METADATA_DIR / 'Stimulation_Info.csv'
filenamesPath = METADATA_DIR / 'Data_file_names.csv'
rightAnswersPath = METADATA_DIR / 'rightAnswers.txt'
if not rightAnswersPath.exists():
    rightAnswersPath = None

# Cached outputs used when the corresponding redo_* flag is False.
P3infoPath = PROCESSED_DATA_DIR / 'p300potentialsinfo.pkl'
behScoresPath = PROCESSED_DATA_DIR / 'behScores.pkl'
prepInfoPath = PROCESSED_DATA_DIR / 'preprocessed_files_info_1.csv'

# Data locations. Place private EEG files under these folders before running.
rawDataPath = RAW_DATA_DIR
prepPath = PREPROCESSED_DATA_DIR
dataPath = prepPath



# Available preprocessed files
files = {}
fileList, measStatus = get_available_measurements(filenamesPath, dataPath)
files['SDR'], files['VDR'] = divide_by_measurement_type(fileList)

# Oddball correspondences
oddDict = {'VDR': 'acoustic',
           'SDR': 'visual'}
#%% PREPROCESS
# ! This sections runs for a several hours if True
manual_rejection = False
preselected_bad_ch = True #To use bad channels selected before by hand as bad
#Using it ensures the reproducibility of the results

if redo_preprocessing:
    # Initiliaze preprocess info storage
    preprocess_info = pd.DataFrame(columns=['measurement', 'task', 'n_bad_ch', 'bad_ch',
                                            'n_task_epochs', 'n_bad_task_epochs',
                                            'n_oddball_epochs', 'n_bad_oddball_epochs',
                                            'path'])
    filenames = [file.name for file in rawDataPath.glob('*raw.fif')]

    for file in filenames:
        if preselected_bad_ch:
            info = pd.read_csv(prepInfoPath)
            idx = info.loc[info['measurement'] == file.replace('raw.fif', '')].index[0]
            bad_ch = ast.literal_eval(info.at[idx, 'bad_ch'])
        else:
            bad_ch = None
        prep_info = preprocess(rawDataPath, file, prepPath, stimInfoPath,
                               bad_ch = bad_ch, manual = manual_rejection)

        preprocess_info = pd.concat(
            [preprocess_info, pd.DataFrame([prep_info])],
            ignore_index=True
        )

    preprocess_info.to_csv(PROCESSED_DATA_DIR / 'preprocessingInfo.csv', index=False)

    dataPath = prepPath






#%% BEHAVIORAL ANALYSIS
# Raw evaluation
# ! This sections runs for a couple of minutes if True

#Results stored for each trial (not measurement!)
if redo_behavioral:
    swapper = np.array([2, 3, 4, 5, 6, 7, 8, 1]) # For button label correction
    sdrScoreAll = evaluate_beh_patients(files['SDR'], 'SDR', rightAnswersPath, swapper)
    #outcome dictionaries contain dictionaries for each patient with the time points and trial-wise accuracy and RT
    vdrScoreAll = evaluate_beh_patients(files['VDR'], 'VDR', rightAnswersPath)
    vobScoreAll = evaluate_beh_patients(files['SDR'], 'oddball')
    aobScoreAll = evaluate_beh_patients(files['VDR'], 'oddball')

    #These results are already saved for whole measurement
    sensitivityVobAll = evaluate_beh_patients(files['SDR'], 'sensitivity')
    sensitivityAobAll =  evaluate_beh_patients(files['VDR'], 'sensitivity')

    # Compute measurement total results
    sumResultsVdr = summarize_all_results(vdrScoreAll)
    sumResultsSdr = summarize_all_results(sdrScoreAll, sdr=True)
    sumResultsVob = summarize_all_results(vobScoreAll)
    sumResultsAob = summarize_all_results(aobScoreAll)

    behScores = {'VDR': sumResultsVdr,
                 'SDR': sumResultsSdr,
                 'vob': sumResultsVob,
                 'aob': sumResultsAob,
                 'vob_s': sensitivityVobAll,
                 'aob_s': sensitivityAobAll}

    #Save data
    behPath = PROCESSED_DATA_DIR / 'behScores.pkl'
    with behPath.open('wb') as file:
        pkl.dump(behScores, file)

else:
    behPath = behScoresPath
    with behPath.open("rb") as file:
        behScores = pkl.load(file)
    sumResultsVdr, sumResultsSdr, sumResultsVob, sumResultsAob,\
        sensitivityVobAll, sensitivityAobAll = behScores.values()





#%% EXTRACTION P300 FEATURES
# ! This sections runs for a couple of minutes if True

# to run the test and visualize channels chosen
# It takes time because all V0 epochs are read and concatenated
if redo_permutation_test:
    #Here only for visualization purposes, not to actually change cluster chosen
    #TODO: Does not work yet
    for task in ['VDR', 'SDR']:
        obs, clust, cluster_pv, H0 = epochs_permutation_test(files[task], oddDict[task], clustInfo=clustInfo)

##note that adj matrix has not been specified for the cluster perm, so interpretation is difficult

clustInfo = {'acoustic': {'cluster': 46, 'chans': ['Cz','CPz','Pz','POz']},
             'visual': {'cluster': 69, 'chans': ['Cz','CPz','Pz','POz']}}

# dictionary clustInfo (imported from physio_features) has channels of the clusters
# previously selected
if redo_P300:
    P3info = {}
    features = ['peak', 'latency', 'amplitude']

    for file in fileList:
        eFile = file.replace('_p_raw', '_oe_epo')
        task = 'VDR' if 'VDR' in eFile else 'SDR'
        epochs = mne.read_epochs(eFile, preload=True).apply_baseline()
        epochs.pick_channels(clustInfo[oddDict[task]]['chans'])
        #TODO: this picks channels that belong to any cluster, maybe make it a bit more specific?
        P3info[eFile] = {}
        for k in epochs.event_id.keys():
            #print(k)
            if len(epochs[k])>0:

                P3info[eFile][k] = {f:v for f,v in zip(features, get_P300(epochs[k]))}

    P3Path = PROCESSED_DATA_DIR / 'p300potentialsinfo.pkl'
    with P3Path.open('wb') as file:
        pkl.dump(P3info, file)


else:
    P3Path = P3infoPath
    with P3infoPath.open("rb") as file:
        P3info = pkl.load(file)

#%% EXTRACTION TIME-FREQUENCY FEATURES

t_window = 4 #seconds before and after cue for the epochs
freqBands = {'delta (1-3)': (1, 4),
              'theta (4-8)': (4, 9),
              'alpha (9-14)': (9, 15),
              'beta (15-30)': (15, 31),
              'total (1-30)': (1,31)}

baseline = {'VDR': (1.5, 3.5),
            'SDR': (1.5,3.5)}

TFPath = PROCESSED_DATA_DIR / 'timefrequencydata.pkl'

tfaInfo = {}

if redo_timefrequency:
    for file in fileList:
        filename = Path(file).name
        print(f'running TFA for {filename}')
        task = 'VDR' if 'VDR' in file else 'SDR'
        data = mne.io.read_raw_fif(file, preload=True)
        epochs = tfa_epocher(data, t_window=t_window)
        f, t, Sxx, BL, tfband = tf_data(epochs, freqBands=freqBands, log_transform=True,
                                              baseline=baseline[task])

        tfaInfo[filename] = tfband
        print(f'current keys in tfaInfo: {tfaInfo.keys()}')
        tfaData = {'Sxx':{}, 'BL': {}}
        tfaData['Sxx'][filename] = Sxx
        tfaData['BL'][filename] = BL

        # sxxpath = f'{file}_spectrum.pkl' #tfaspectra will contain the raw spectra for all channels
        # file = open(sxxpath, 'wb')
        # pkl.dump(Sxx, file)
        # file.close()
        TFpath = PROCESSED_DATA_DIR / f'{filename}_timefrequencydata_raw.pkl'
        with TFpath.open('wb') as file:
            pkl.dump(tfaData, file)

        with TFPath.open('wb') as file:
            pkl.dump(tfaInfo, file)
        print(f'dumped tfaInfo with keys: {tfaInfo.keys()}')
else:
    with TFPath.open('rb') as file:
        tfaInfo=pkl.load(file)



# else:
#     with open(TFPath, "rb") as pf:
#         tfaInfo = pkl.load(pf)

##read in one file to get time and frequency indices
##to save memory, they can also be generated as simple arrays
epochs = tfa_epocher(mne.io.read_raw_fif(fileList[0], preload=True), t_window=t_window)
f, t, _, _, _ = tf_data(epochs, freqBands=freqBands, log_transform=True,
                                      baseline=(0,0) )

f = np.arange(1,31)
t = np.linspace(-3.5,3.5,num=141)
#extract raw spectra and relative spectra, plot along with topos
over_time_abs = np.nan*np.zeros((len(fileList),len(f),len(t)))
over_time_rel = over_time_abs.copy()

over_chans_abs = np.nan*np.zeros((len(fileList),len(f), 64))
over_chans_rel = over_chans_abs.copy()
taskvec = np.nan*np.zeros(len(fileList))
#read in single-subject spectra from pickle files
for i,file in enumerate(fileList):

    print(i)
    filename = Path(file).name
    if 'VDR' in filename:
        taskvec[i] = 1
    elif 'SDR' in filename:
        taskvec[i] = 2
    TFpath = PROCESSED_DATA_DIR / f'{filename}_timefrequencydata_raw.pkl'
    with TFpath.open("rb") as pf:
        tfaData = pkl.load(pf)


    Sxx = np.array(list(tfaData['Sxx'].values()))
    rel = Sxx - np.array(list(tfaData['BL'].values()))

    Sxx = np.squeeze(Sxx)
    rel = np.squeeze(rel)
    if len(Sxx.shape) != 4:
        continue

    over_time_abs[i,:,:] = np.median(Sxx, (0,1))
    over_time_rel[i,:,:] = np.median(rel, (0,1))

    over_chans_abs[i,:,:] = np.median(Sxx, (0,3)).T
    over_chans_rel[i,:,:] = np.median(rel, (0,3)).T
    # tfaInfo[filename] = {band: np.nanmedian(rel[:,:,lo:hi,:]) for band, (lo,hi) in freqBands.items()}






#TODO: include correlation topos!

#%% relative spectra in both tasks
fig,ax = plt.subplots(2,2) #for relative spectra in both tasks


im2=ax[0,0].pcolor(np.nanmedian(over_time_rel[taskvec==2],0))
ax[0,0].set_title('Spectral response (SDR)')
cbar=plt.colorbar(im2, ax=ax[1,0])
cbar.set_label('dB', rotation=270)



im=ax[1,0].pcolor(np.nanmedian(over_time_rel[taskvec==1],0))
ax[1,0].set_title('Spectral response (VDR)')
cbar=plt.colorbar(im, ax=ax[0,0])
cbar.set_label('dB', rotation=270)

i=.1
for band, (lo,hi) in freqBands.items():
    ax[1,1].plot(np.nanmedian(over_time_rel[taskvec==1,lo:hi,:],(1,0)),label=band, color=(i,i,.6))
    ax[1,1].legend()
    ax[0,1].plot(np.nanmedian(over_time_rel[taskvec==2,lo:hi,:],(1,0)),label=band, color=(i,i,.6))
    ax[0,1].legend()
    i+=.2



ax[0,1].set_title('Frequency bands (SDR)')

ax[1,1].set_title('Frequency bands (VDR)')


lab='frequency [Hz]'
ax[0,0].set_ylabel(lab)
ax[1,0].set_ylabel(lab)

lab='power [dB]'
ax[0,1].set_ylabel(lab)
ax[1,1].set_ylabel(lab)
lab='time [s]'

for i in range(2):
    for k in range(2):
        ax[i,k].set_xticks(range(len(t))[::10])
        ax[i,k].set_xticklabels(t[::10])
        ax[i,k].set_xlabel(lab)

fig.set_size_inches(12,6)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'spectral_response.svg')
#%% absolute spectrum
plt.figure()
plt.plot(np.nanmedian(over_time_abs, (0,2)))

#%% relative spectrum topo
from multichannel_tools.viz import my_topomap
fig=plt.figure()
i=0
clim = {
        'delta (1-3)':[0,1.5],
        'theta (4-8)':[-.5,.5],
        'alpha (9-14)':[-.4,.4],
        'beta (15-30)':[-.3,.3]}
for task,taski in {'SDR':2, 'VDR':1}.items():
    for band, (lo,hi) in freqBands.items():
        if band == 'total (1-30)':
            continue
        i+=1
        clo,chi = clim[band]
        plt.subplot(2,4,i)
        plt.title(f'{task}, {band}')
        my_topomap(np.nanmedian(over_chans_rel[taskvec==taski,lo:hi,:], (0,1) ), chans = epochs.info.ch_names, clim=(clo,chi))
plt.gcf().set_size_inches(14,5)
plt.savefig(FIGURES_DIR / 'spectral_response_topos.svg')







#%% ALL FEATURES TOGETHER FOR STATISTICS
# Build-up the master DataFrame with all variables by measurement

#Grouping by stimulus type
oddCategory = {'circle':'nontarget',
               'lowPitch': 'nontarget',
               'large': 'target',
               'largeCircle': 'target',
               'highPitch': 'target',
               'distractor': 'distractor'}

varsDict = {'info': ['ID', 'stimulation', 'task', 'time'],
            'physio': ['P3distractoramplitude', 'P3distractorlatency',
                       'P3nontargetamplitude', 'P3nontargetlatency', 'P3targetamplitude',
                       'P3targetlatency'],#, 'TFalpha', 'TFbeta', 'TFdelta', 'TFtheta', 'TFtotal'],
            'timefreq': ['TFalpha', 'TFbeta', 'TFdelta', 'TFtheta', 'TFtotal'],
            'P3potentials' : ['P3distractoramplitude', 'P3distractorlatency',
                              'P3nontargetamplitude', 'P3nontargetlatency',
                              'P3targetamplitude', 'P3targetlatency'],
            'behavior': ['oddballAcc', 'oddballRT', 'sensitivity', 'taskAcc', 'taskRT']}

varsDict['physio'].extend(['TF'+k for k in freqBands.keys()])

# Grand dict with all data
pats = list(measStatus['ID'])
measurements = list(measStatus.columns.delete(0))
info = {}
for p in pats:
    for m in measurements:
        if measStatus.loc[measStatus['ID'] == p][m].item():
            measurement = str(p) + '_' + m
            info[measurement] = {}
            #info: stim, task, time, ID
            info[measurement]['ID'] = p
            info[measurement]['stimulation'] = get_stim_info(stimInfoPath,
                                                             measurement[:-4])
            info[measurement]['stimNum'] = 0 if info[measurement]['stimulation'] == 'OFF' else 1
            info[measurement]['task'] = m[-3:]
            info[measurement]['time'] = m[:-4]

            #results: acc ratio, response time, oddball acc, oddball rt, sensitivity idx
            if 'VDR' in m:
                task, oddball, sensitivity = sumResultsVdr, sumResultsAob, sensitivityAobAll
            else:
                 task, oddball, sensitivity = sumResultsSdr, sumResultsVob, sensitivityVobAll

            if m in task[str(p)].keys():
                info[measurement]['taskAcc'] = task[str(p)][m][0]
                info[measurement]['taskRT'] = task[str(p)][m][1]
                info[measurement]['oddballAcc'] = oddball[str(p)][m][0]
                info[measurement]['oddballRT'] = oddball[str(p)][m][1]
                info[measurement]['sensitivity'] = sensitivity[str(p)][m][0]

                #physio features 1: p3 latency, p3 amplitude * stimuli
                p3data = P3info[[key for key in P3info.keys() if measurement in key][0]]
                for s in p3data: #stimulus type
                    for f in p3data[s]: #feature
                        cat = oddCategory[s]
                        info[measurement]['P3'+cat+f] = p3data[s][f]

                #physio features 2: delta, theta, alpha, beta & total after cue
                tfdata = tfaInfo[[key for key in tfaInfo.keys() if measurement in key][0]]
                for b in tfdata:
                    info[measurement]['TF'+b] = tfdata[b]

# Convert to pandas data frame
infoDf = pd.DataFrame(info.values())

columnsNorm = varsDict['physio'] + varsDict['behavior']

infoDf = add_norm_values(infoDf, columns=columnsNorm)

# Save
infoDf.to_csv(PROCESSED_DATA_DIR / 'infoDf.csv', index =False)

#Do subsets according to analizable patients and save for analysis in R
subsetInfo = pd.read_csv(subsetInfoPath, index_col=False)
for ta in ['SDR', 'VDR']:
    for ti in ['short', 'long', 'all']:
        df = subsetter(infoDf, task = ta, time = ti, analizableDf = subsetInfo)
        df.to_csv(R_INPUT_DIR / f'{ta}_{ti}.csv', index=False)
#
#
# # -> R : Run statistics.R saved in the same directory as this script
#
# #%% STATISTICS: FROM R
# # Read what was run in R and convert to be used in the report
# ttests = pd.read_csv(STATISTICS_RESULTS_DIR / 'tTestsResults.csv')
# ttests = ttests.drop(0, axis=0).drop('Unnamed: 0', axis=1)
# modelsResults = pd.read_csv(STATISTICS_RESULTS_DIR / 'GLMMresults.csv')
# modelsResults = modelsResults.drop(0, axis=0).drop('Unnamed: 0', axis=1)
#
# #convert to DF with significance info
# sigttests = significance_df(ttests)
# sigmodels = significance_df(modelsResults)
#
# # Save for projects' documentation
# sigttests.to_csv(STATISTICS_RESULTS_DIR / 'tTestsResultsSign.csv', index = False)
# sigmodels.to_csv(STATISTICS_RESULTS_DIR / 'GLMMresultsSign.csv', index = False)

#%% STATISTICS: CORRELATION BETWEEN PHYSIO AND BEHAVIORAL

# Note : The data frames generated here compute correlation between all behavioral
# and all physiological values of a measurement. In the documentation and data
# interpretation only connected variables (time-frequency to VDR and SDR,
# P3 to oddball) were considered

#Generate correlation values between physio and behavioral data
drop_columns = ['ID', 'stimulation', 'time', 'task']
for t in ['short', 'long']:
    for task in ['SDR', 'VDR']:
        data = subsetter(infoDf, task=task, time=t, analizableDf = subsetInfo)
        corrDf = corr_df(data, varsDict['behavior'], varsDict['physio'], drop_columns=drop_columns)
        corrDf = significance_df(corrDf)
        corrDf.to_csv(STATISTICS_RESULTS_DIR / f'Corr{task}{t}.csv' )

# Generate correlation values between long-term cognitive change and physio values
for task in ['SDR', 'VDR']:
        data = subsetter(infoDf, task=task, time='long', analizableDf = subsetInfo)
        data = data[data['time'] != 'V0']
        colnames = [n + 'Norm' for n in varsDict['behavior']]
        corrDf = corr_df(data, colnames, varsDict['physio'], drop_columns=drop_columns)
        corrDf = significance_df(corrDf)
        corrDf.to_csv(STATISTICS_RESULTS_DIR / f'DiffsCorr{task}long.csv' )




#TODO: add correlation topo here!!
# Generate and store correlation values between baseline values and difference values from V4/V5
infoDf = add_baseline_values(infoDf, columnsNorm)
drop_columns = ['ID', 'stimulation', 'time', 'task']
EEGcors_topo = {}
for task in ['SDR', 'VDR']:
    EEGcors_topo[task]={}
    for stim in ['OMNI', 'DIR']:
            data = subsetter(infoDf, task=task, time='long', analizableDf = subsetInfo)
            data = data[data['time'] != 'V0']
            data = data[data['stimulation'] == stim]
            rownames = [n + 'BL' for n in varsDict['physio']]
            colnames = [n + 'Norm' for n in varsDict['behavior']]
            corrDf = corr_df(data, colnames, rownames, drop_columns=drop_columns)
            corrDf = significance_df(corrDf)
            corrDf.to_csv(STATISTICS_RESULTS_DIR / f'BLNormCorr{task}{stim}.csv' )
            
            #select elements in eeg data corresponding to current task and IDs
            sel = np.isin(infoDf.ID,data.ID)&np.isin(infoDf.time, ['V4','V5'])&(infoDf.task==task)
            if (stim=="DIR"):
                sel&=infoDf.ID!=0 #exclude patient 0 because of excess power
            for band, (lo,hi) in freqBands.items():
                accdat = infoDf.taskAccNorm.to_numpy()[sel]
                rtdat =  infoDf.taskRTNorm.to_numpy()[sel]
                banddat = np.median(over_chans_rel[sel,lo:hi,:],1)
                EEGcors_topo[task][f'Acc_{band}'] = [stats.pearsonr(accdat, banddat[:,chan])[0] for chan in range(banddat.shape[1])]
                EEGcors_topo[task][f'RT_{band}'] = [stats.pearsonr(rtdat, banddat[:,chan])[0] for chan in range(banddat.shape[1])]

#%% VISUALIZATIONS TO REPORT
# Descriptive table for baseline behavioral results.

#Table with average behavioral results at baseline
behInfoDf = infoDf[['ID', 'time','task','taskAcc', 'taskRT', 'oddballAcc', 'oddballRT', 'sensitivity']]
behInfoV0 = pd.DataFrame(index = ['SDR', 'VDR', 'auditory oddball', 'visual oddball'],
                         columns = ['Accuracy mean', 'Accuracy SD', 'Response time mean',
                                    'Response time SD', 'Sensitivity mean',
                                    'Sensitivity SD'])
dc = {'SDR': 'visual oddball',
      'VDR':'auditory oddball'}
df = behInfoDf[behInfoDf['time'] == 'V0']
for task in ['SDR', 'VDR']:
    df = behInfoDf[behInfoDf['time'] == 'V0']
    df = df[df['task'] == task]
    behInfoV0.loc[(task, 'Accuracy mean')] = df['taskAcc'].mean()
    behInfoV0.loc[(task, 'Accuracy SD')] = df['taskAcc'].std()
    behInfoV0.loc[(task, 'Response time mean')] = df['taskRT'].mean()
    behInfoV0.loc[(task, 'Response time SD')] = df['taskRT'].std()
    behInfoV0.loc[(dc[task], 'Accuracy mean')] = df['oddballAcc'].mean()
    behInfoV0.loc[(dc[task], 'Accuracy SD')] = df['oddballAcc'].std()
    behInfoV0.loc[(dc[task], 'Response time mean')] = df['oddballRT'].mean()
    behInfoV0.loc[(dc[task], 'Response time SD')] = df['oddballRT'].std()
    behInfoV0.loc[(dc[task], 'Sensitivity mean')] = df['sensitivity'].mean()
    behInfoV0.loc[(dc[task], 'Sensitivity SD')] = df['sensitivity'].std()

behInfoV0.to_csv(STATISTICS_RESULTS_DIR / 'baselinebehavioralresults.csv', index=False)
#
# #BOXPLOTS
#
# data = infoDf
# depvar = ['taskAccNorm', 'taskAccNorm', 'TFalpha', 'TFbeta', 'TFbeta', 'TFdelta',
#           'TFbeta', 'P3distractoramplitude', 'P3distractoramplitude']
# indvar = ['stimulation', 'stimulation', 'time', 'time', 'stimulation', 'stimulation',
#           'time', 'stimulation', 'stimulation']
# hue = [None, None, 'stimulation', 'stimulation', None, None, 'stimulation',
#        None, None]
# task = ['VDR', 'SDR', 'VDR', 'VDR', 'VDR', 'VDR', 'SDR', 'VDR', 'SDR']
# time = ['long', 'long', 'short', 'short', 'long', 'long', 'long', 'short', 'short']
# for i in range(len(depvar)):
#     plot_for_stats(data, depvar[i], indvar[i], hue=hue[i], task=task[i],
#                    time=time[i], analizableDf=subsetInfo)
#
# #
# if plotP3waves:
#     tasks = ['VDR', 'SDR']
#     for task in tasks:
#         filesV3 = [f for f in files[task] if 'V3' in f]
#         epochsdistV3 = {}
#         for f in filesV3:
#             measurement = f.split('/')[-1][:-14]
#             pat = int(measurement.split('_')[0])
#             if subsetInfo[subsetInfo['ID'] == pat][task+'_short'].item():
#                 eFile = f.replace('_p_raw', '_oe_epo')
#                 epochs = mne.read_epochs(eFile, preload=True).apply_baseline()
#                 epochs.pick_channels(clustInfo[oddDict[task]]['chans'])
#                 epochs = epochs['distractor'].apply_baseline()\
#                     .get_data().mean(axis=0).std(axis=0)
#                 stim = get_stim_info(stimInfoPath, measurement)
#                 if stim not in epochsdistV3:
#                     epochsdistV3[stim] = epochs
#                 else:
#                     epochsdistV3[stim] = np.dstack((epochsdistV3[stim], epochs))
#
#         plt.figure()
#         colors=['mediumblue','limegreen' , 'deepskyblue','aquamarine', 'mediumblue', 'aquamarine', 'yellowgreen']
#         for i, stim in enumerate(['OFF', 'DIR', 'OMNI']):
#             array = epochsdistV3[stim].squeeze().mean(axis=1)
#             bl = array[0:1000].mean()
#             array = array - bl
#             error = epochsdistV3[stim].squeeze().std(axis=1)/np.sqrt(epochsdistV3[stim].shape[2])
#             time = np.linspace(-0.2, 1.0, 6001)
#             plt.plot(time,array, label = stim, color = colors[i])
#             plt.fill_between(time, array-error, array + error, alpha = 0.3, color=colors[i])
#
#         plt.ylabel(renamer('P3distractoramplitude', task))
#         plt.xlabel('time [s]')
#         plt.legend()
#         sns.despine()
#
# # SCATTER PLOTS
# # In different colors :)
#
# #Part 1[]
# task = ['VDR', 'VDR']
# time = ['short', 'long']
# x=['TFalpha', 'TFtheta']
# y=['taskAcc', 'taskAcc']
# #stim = 'OMNI'
#
# color = ['deepskyblue','limegreen' ,'aquamarine', 'yellowgreen', 'mediumblue']
# for i in range(len(task)):
#     data = subsetter(infoDf, task=task[i], time=time[i], analizableDf=subsetInfo)
#     plt.figure()
#     plt.scatter(data[x[i]],data[y[i]], color=color[0])
#     plt.xlabel(renamer(x[i], task[i]))
#     plt.ylabel(renamer(y[i], task[i]))
#     sns.despine()
#
# #Part 2 (changes)
# task = 'VDR'
# time = 'long'
# x = 'TFbeta'
# y = 'taskAccNorm'
# data = subsetter(infoDf, task=task, time=time, analizableDf=subsetInfo)
#
# data = data[data['time'] != 'V0']
# plt.figure()
# plt.scatter(data[x],data[y], color=color[1])
# plt.xlabel(renamer(x, task))
# plt.ylabel(renamer(y, task))
# sns.despine()
#
# #Part 3 (baseline -> change)
#
# task = [ 'VDR', 'SDR']
# time = [ 'long', 'long']
# x=['TFbetaBL', 'TFalphaBL']
# y=[ 'taskAccNorm', 'taskAccNorm']
# stim = ['OMNI', 'DIR']
#
# for i in range(len(task)):
#     data = subsetter(infoDf, task=task[i], time=time[i], analizableDf=subsetInfo)
#     data = data[data['stimulation'] == stim[i]]
#     data = data[data['time'] != 'V0']
#     plt.figure()
#     plt.scatter(data[x[i]],data[y[i]], color=color[2])
#     plt.xlabel(renamer(x[i], task[i]))
#     plt.ylabel(renamer(y[i], task[i]))
#     sns.despine()

plt.figure()
plt.subplot(2,2,1)
plt.title('SDR / Accuracy')
my_topomap(np.array(EEGcors_topo['SDR']['Acc_alpha (9-14)']), chans = epochs.ch_names)

plt.subplot(2,2,2)
plt.title('SDR / RT')
my_topomap(np.array(EEGcors_topo['SDR']['RT_alpha (9-14)']), chans = epochs.ch_names)


plt.subplot(2,2,3)
plt.title('VDR / Accuracy')
my_topomap(np.array(EEGcors_topo['VDR']['Acc_alpha (9-14)']), chans = epochs.ch_names)

plt.subplot(2,2,4)
plt.title('VDR / RT')
my_topomap(np.array(EEGcors_topo['VDR']['RT_alpha (9-14)']), chans = epochs.ch_names)

plt.savefig(FIGURES_DIR / 'EEG_cor_topos.png')
