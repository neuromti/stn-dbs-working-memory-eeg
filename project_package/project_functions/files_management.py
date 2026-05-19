# -*- coding: utf-8 -*-
"""File discovery and measurement-management helpers."""

from pathlib import Path

import pandas as pd
import mne
import numpy as np


def automation_filenames(namesPath, dataPath, savePath, patients='all', measurements = 'all'):
    '''
    Since EEG recordings were saved with unsystematized names, this function
    converts all .vhdr & Co. files into .fif files with automated names that 
    contains patient, measurement and task type info in the file name. It requires 
    that the data is saved in a path .../[ID]/[MEASUREMENT]/[file]
    It is thought to be feeded with a .csv where the single filenames are 
    documented. For the VDR and SDR measurements a 'Data_file_names.csv'
    should be saved in this project's folder.
    
    It returns a list with paths to the new files and the paths that were not found.
    Parameters
    ----------
    namesPath : str
        path of the csv file with all filenames. It has to have a column called
        'ID' with subjects' IDs. The other columns need to be the names of the 
        measurements.
    dataPath : str
        Path to the folder were data is stored. In this folder there should be
        folders with the subjects' IDs as name.
    savePath : str
        path where all converted file will be stored.
    patients : list, optional
        List with subset of patients, whose info is to be converted. If 'all'
        all IDs in the file (given by namesPath) are considered. The default is 'all'.
    measurements : list, optional
        List with subset of measurement to be converted. The names need to correspond to
        columns in the file with the filenames. If 'all' all columns in the file
        are considered. The default is 'all'.
    Returns
    -------
    namesData : list
        List with all the new paths.
    failedFiles : list
        List with all the files that were not found.

    '''
    dataPath = Path(dataPath)
    savePath = Path(savePath)
    fileNames = pd.read_csv(namesPath, index_col='ID')
    namesData = []
    failedFiles = []
    
    # Create subjects and measurements subsets if required
    if patients != 'all':
        fileNames.drop([idx for idx in fileNames.index if idx not in patients], inplace=True)
    if measurements != 'all':
        fileNames.drop([col for col in fileNames.columns if col not in measurements], axis=1, inplace=True)

    for pat in fileNames.index:
        for meas in fileNames.columns:
            patient = str(int(pat))
            measurement = str(meas[0:2]) #Take only the "Vi" part of name to find file
            name = str(fileNames.loc[pat][meas])
            path = dataPath / patient / measurement / f'{name}.vhdr'
            if path.exists():
                data = mne.io.read_raw_brainvision(path)
                out_path = savePath / f'{patient}_{meas}raw.fif'
                data.save(out_path, overwrite=True)
                namesData.append(str(out_path))
            else: 
                failedFiles.append(str(path))
    return namesData, failedFiles




def get_available_measurements(dfPath, dirPath, savePath=None):
    '''
    This function is used to get information about the successful preprocessed 
    measurements. It bases its search in the existence of the expected preprocessed
    files (a main file and two epochs' files).
           
    The output list with all filenames facilitated the iteration over all
    preprocessed files.

    Parameters
    ----------
    dfPath : str
        path to .csv with filenames. It is the same that is feeded to 
        automation_filenames. It serves as template for the expected preprocessed
        files. 
    dirPath : str
        path of the directory where all preprocessed files are saved.
    savePath : str, optional
        path to which the information of available files (output fileNames)
        should be saved. If None the list is not saved. The default is None.

    Returns
    -------
    fileList : list
        list with preprocessed available files (only main file).
    measStatus : pandas DataFrame
        a data frame in the same structure as the one found by dfPath with boolean
        values with True if the measurement is considered available.

    '''
    dirPath = Path(dirPath)
    fileList = []
    
    #These ends for the filenames correspond to the automated assigning during the
    #preprocessing pipeline
    pathEnd = ['_p_raw.fif', '_oe_epo.fif', '_te_epo.fif']
    modelDf = pd.read_csv(dfPath)
    measStatus = pd.DataFrame(columns=modelDf.columns)
    measStatus.ID = modelDf.ID
    
    for patient in measStatus.ID:
        # It starts by 1 as the ID columns is the first
        for m in measStatus.columns[1:]:
            #Define path begin
            expected_paths = [dirPath / f'{patient}_{m}{end}' for end in pathEnd]
            idx = measStatus[measStatus.ID == patient].index[0]
            
            # If the preprocessing was successful 
            if all(path.exists() for path in expected_paths):
                    measStatus.loc[idx, m] = True
                    fileList.append(str(expected_paths[0]))
            else: 
                    measStatus.loc[idx, m] = False
            if savePath:
                Path(savePath).write_text('\n'.join(fileList))
    return fileList, measStatus
def duration_measurements(df, dirPath, sfreq=5000, savePath=None):
    '''
    This function stores the duration in seconds of a set of recordings. As a
    template for the info storage it takes a DataFrame with IDs and measurements.
    It can be the one that is read by automation_filenames.
    
    The outputted DataFrame can be a useful information to identify recordings
    that were interrupted early.

    Parameters
    ----------
    df : pandas DataFrame
        Template data frame with an 'ID' column with the subjetcs, whose measurements
        are to be considered. All other columns are interpreted as the measurements
        names to be looked for.
    dirPath : str
        path to directory where the measurement files are stored. All files need to be 
        in .fif format
    sfreq : int, optional
        Sampling frequency. The default is 5000.
    savePath : str, optional
        path to save the resulting data frame as .csv file. If None nothing is
        saved. The default is None.

    Returns
    -------
    durationAll pandas DataFrame
        data frame with duration of all measurements.

    '''
    dirPath = Path(dirPath)
    durationAll = pd.DataFrame(columns=df.columns, index=df.ID).drop('ID', axis=1)
    #This end is based on the automated naming during the preprocessing pipeline
    end = '_p_raw.fif'
    
    for pat in durationAll.index:
        for col in durationAll.columns:
            # Try to open only if the file exists 
            data_file = dirPath / f'{pat}_{col}{end}'
            if data_file.exists() and df.loc[df.ID == pat][col].bool():
                measurement = mne.io.read_raw_fif(data_file)
                durationAll.loc[pat][col] = measurement.__len__()/sfreq
            # Assign nan so the whole DataFrame can be handled with Numpy 
            else:
                durationAll.loc[pat][col] = np.nan
                
    if savePath:
        durationAll.to_csv(Path(savePath) / 'durationAllMeasurements.csv')
    return durationAll

def divide_by_measurement_type(fileList):
    '''
    Simple function to divide filenames by the type of task being performed
    during the recording. Its classical input is the list ouputted by the 
    get_available_measurements function

    Parameters
    ----------
    fileList : list of str
        list with filenames or filepaths.

    Returns
    -------
    sdrFiles : list of str
        list with filenames corresponding to the SDR task.
    vdrFiles : list of str
        list of filenames corresponding to the VDR task.

    '''
    vdrFiles = [file for file in fileList if 'VDR' in file]
    sdrFiles = [file for file in fileList if 'SDR' in file]
    return sdrFiles, vdrFiles
