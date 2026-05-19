# -*- coding: utf-8 -*-
"""Behavioral scoring utilities for the working-memory EEG analysis."""

import ast
from pathlib import Path

import pandas as pd
import mne
import numpy as np
import scipy.stats as st
#%%

    
# From source code (config files for measurements)
sdr_default_V0 = [[3, 5], [4, 3], [1, 3], [5, 0], 
                  [7, 0], [2, 7], [2, 6], [6, 0], 
                  [1, 7], [7, 2], [5, 1], [3, 4], 
                  [7, 1], [5, 7], [0, 5], [0, 1], 
                  [1, 2], [0, 5], [6, 4], [7,3]]
sdr_default_V3 = [[3, 1], [5, 7], [0, 2], [4, 0], 
                  [3, 7], [0, 3], [0, 7], [7, 3], 
                  [2, 4], [5, 5], [2, 1], [1, 3], 
                  [7, 3], [7, 5], [4, 3], [1, 1], 
                  [4, 4], [7, 3], [2, 5], [0, 5]]
sdr_default_V4 =[[1, 7], [5, 2], [4, 7], [3, 7], 
                 [7, 4], [6, 3], [1, 7], [7, 4], 
                 [1, 4], [3, 5], [5, 6], [4, 7], 
                 [2, 4], [3, 1], [3, 1], [0, 4], 
                 [3, 4], [1, 3], [0, 6], [4, 6]]
sdr_default_V5 = [[3, 1], [1, 2], [0, 7], [6, 3], 
                  [6, 0], [3, 5], [0, 5], [6, 5], 
                  [3, 6], [1, 0], [1, 4], [5, 2], 
                  [0, 1], [1, 3], [3, 1], [3, 7], 
                  [2, 7], [7, 0], [4, 3], [7, 2]] 
vdr_default_V0 = [[0, 1], [0, 1], [1, 0], [0, 1], [0, 1],
                  [0, 1], [0, 1], [1, 0], [1, 0], [1, 0],
                  [0, 1], [1, 0], [1, 0], [1, 0], [0, 1],
                  [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]
vdr_default_V3 = [[1, 0], [1, 0], [0, 1], [1, 0], [0, 1],
                  [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                  [0, 1], [0, 1], [1, 0], [0, 1], [0, 1],
                  [0, 1], [0, 1], [1, 0], [1, 0], [0, 1]]
vdr_default_V4 = [[0, 1], [1, 0], [1, 0], [1, 0], [0, 1],
                  [1, 0], [0, 1], [1, 0], [1, 0], [1, 0],
                  [0, 1], [0, 1], [0, 1], [0, 1], [1, 0],
                  [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]]
vdr_default_V5 = [[0, 1], [1, 0], [0, 1], [1, 0], [1, 0],
                  [1, 0], [0, 1], [1, 0], [1, 0], [0, 1],
                  [0, 1], [1, 0], [1, 0], [1, 0], [0, 1],
                  [0, 1], [1, 0], [0, 1], [1, 0], [1, 0]]

DEFAULT_RIGHT_ANSWERS = {
    'SDR': {'V0': sdr_default_V0, 'V3': sdr_default_V3,
            'V4': sdr_default_V4, 'V5': sdr_default_V5},
    'VDR': {'V0': vdr_default_V0, 'V3': vdr_default_V3,
            'V4': vdr_default_V4, 'V5': vdr_default_V5},
}


def write_default_right_answers(path):
    """Write default task-answer configuration to disk if an external file is needed."""
    Path(path).write_text(str(DEFAULT_RIGHT_ANSWERS))

#%%
def find_right_answers(measurement, measurementType, path=None):
    '''
    It gets the correct answers for a given measurement

    Parameters
    ----------
    measurement : str
        Measurement's ID: V0, V3, V4 or V5.
    measurementType : str
        'SDR' or 'VDR'.
    path : str, optional
        Path of txt file where answers are stored. If omitted, built-in task
        defaults are used.

    Returns
    -------
    answers: list of lists of int
        The correct positions, for each trial a list inside the outputted list.

    '''
    if path is None:
        rightAnswers = DEFAULT_RIGHT_ANSWERS
    else:
        rightAnswers = ast.literal_eval(Path(path).read_text())
    return rightAnswers[measurementType][measurement]

def markers_fixer(annotations):
    '''
    For button pressing markers that are result of overlap between the hardware
    error S128 and an actual button pressing this function fixes the markers to
    have the label of only the pressed button.
    
    Further explanation: faulty labels are assumed to be generated as:
        S128 + S[actual pressed button label].
        As the maximal button value was 8, the faulty labels are those with a number
        from 129 to 136

    Parameters
    ----------
    annotations : mne Annotations object
        annotations of the measurement where markers should be fixed.

    Returns
    -------
    annotations : mne Annotations object
        annotations with the corrected labels (event_id).

    '''
    
    markers = annotations.description 
    errors = np.arange(129,137).astype(str)

    for i, marker in enumerate(markers):
        if np.sum([1 for e in errors if e in marker]):
            #Extract integer to be able to substract the 128 error
            error = int([e for e in errors if e in marker][0])
            markers[i] = markers[i].replace(str(error), str(error-128))
    # Replaced description with fixed labels
    annotations.description = markers
    
    print('Marker errors were fixed.')
    return annotations


def evaluate_VDR(filePath, rightAnswersPath, swapper=None):
    '''
    Evaluates the SDR task for a single measurement. It is a trial by trial 
    evaluation. It compares the markers in annotations with the programmed
    markers. For successful trials it computes the response time too.
    
    Information is stored in dictionary with information of all trials.

    Note: during the present analysis it was observed that the positions in the
    code did not correspond to the button markers as 0 -> 1 and 1 -> 2, but
    0 -> 2 and 1 -> 1. This is why all markers first (1) brought to be named 1 
    and 2 nd then during the comparison swapped (2).
    
    Parameters
    ----------
    filePath : str
        path to .fif file with the measurement to be evaluated.
    rightAnswersPath : str
        path to .txt file with dictionary with the right answer.
    swapper : None
        has no meaning in this function. It is put as parameter for 
        compatibility with the evaluate_SDR function in the evaluate_beh_patients
        function

    Returns
    -------
    vdrScores : dict
        trial-by-trial scores obtained in the VDR measurement. Trial indexes
        are the keys for list of length 2. Position 0 has boolean value for 
        accuracy, position 1 has response time value if accuarcy is True. 
        In the contrary case it is nan. 

    '''
    annotations = mne.read_annotations(filePath)
    annotations = markers_fixer(annotations)

    #Find right answers
    measurement = filePath[filePath.find('V'):filePath.find('V')+2]
    rightPos = find_right_answers(measurement, 'VDR', rightAnswersPath) 
    
    #Select annotations of (valid) button pressings
    mask = np.logical_and(['Stimulus' in cue for cue in annotations.description],
                          [not 'S128' in cue for cue in annotations.description])
    
        
    target = 'highPitch'
    askIdx = -1
    vdrScores = {}
    for i in range(annotations.__len__()):
        # Look for ask/query cues
        if 'ask' in annotations[i]['description']:
            print('\n ---------------- \n')
            print('Ask cue at ', annotations[i]['onset'], 's')
            askIdx += 1
            
            #Check if there are more (valid) button cues after asking
            if sum(mask[i:]):
                # The +1 'aligns' the porgrammed positions (0 or 1) with the
                # recorded positions (1 + 2). (1) in note
                rightAns = int(np.argwhere(rightPos[askIdx]).squeeze()) + 1
                print('Right answer was at position #', rightAns)
                #Get idx of pressed button and compute time to query cue and
                #get the marker of the pressed button
                pressIdx = int(i + np.argwhere(mask[i:])[0].squeeze())
                timeDiff = annotations[pressIdx]['onset'] - annotations[i]['onset']
                pressAns = int(annotations[pressIdx]['description'].replace('Stimulus/S','').strip())
                print('The next button pressed was', pressAns, ',', timeDiff, 
                      's after the cue')
                
                # Check accuracy and validity, store trial score
                # Inequality due to (2) of Note in description 
                if pressAns != rightAns and timeDiff <= 5 and not target in annotations[i:pressIdx].description:
                    vdrScores[askIdx] = (True, timeDiff)
                    print('Correct!')
                else: 
                    vdrScores[askIdx] = (False, np.nan)
                    print('Wrong or invalid')
            else:
                vdrScores[askIdx] = (False, np.nan)
                print('Pressed nothing afterwards')
    return vdrScores

def nums_swapper(length, cut):
    '''
    Swaps numbers in a range parting form a given position.
    
    It was useful to test which of the combinations of programmed markers and
    buttons pressed was the most plausible one. 

    Parameters
    ----------
    length : int
        length of range to be swapped.
    cut : int
        index at which the given series should be cut. This position and the 
        next will be the intial part of new array and the position until the one
        before the cut will be the final part of the new array

    Returns
    -------
    array : Numpy array
        swapped array.

    '''
    basis = np.arange(length)
    return np.concatenate([basis[cut:], basis[:cut]])

def new_num_assigner(givenNums, swapper):
    '''
    It assigns to a given array of numbers new numbers according to the swapper
    given.
    
    It is used to swap the IDs of the buttons pressed to the scale 'aligned'
    to the IDs in the right answers. 

    Parameters
    ----------
    givenNums : Numpy array
        the number to be transformed to the convention given by the swapper.
    swapper : Numpy array
        the numbers to be assigned to the given numbers.

    Returns
    -------
    newNums : list
        array with the new numbers assigned to the givenNums according to the
        swapper.

    '''
    
    # The +1 'aligns' the programmed positions (0 to 7) with the
    # recorded positions (1 to 8). 
    assigner = dict(zip(np.arange(len(swapper))+1, swapper))
    return [assigner[num] for num in givenNums]

      

def evaluate_SDR(filePath, rightAnswersPath, swapper=None):
    '''
    Evaluates the SDR task for a single measurement. It is a trial by trial 
    evaluation. It compares the markers in annotations with the programmed
    markers. For successful trials it computes the response time too.
    
    Information is stored in dictionary with information of all trials.

    Note: during the present analysis it was observed that the positions in the
    code did not correspond to the button markers in a logical order (0 -> 1, 
    1 -> 2, etc). This is why the swapper is used. It should correspond to the 
    right assignation according to the tests made. 

    Parameters
    ----------
    filePath : str
        path to .fif file with the measurement to be evaluated.
    rightAnswersPath : str
        path to .txt file with dictionary with the right answer.
    swapper : Numpy array or None.
        array with the right button pressing IDs organized in the way that the 
        first corresponds to the recorded marked as 1, and so on. If None, the IDs
        of the recorded button pressing are kept unchanged

    Returns
    -------
    sdrScores : dict
        trial-by-trial scores obtained in the SDR measurement. Trial indexes
        are the keys for tuple of length 2. Position 0 number of right pressed 
        positions (0/1/2), position 1 has response time value if accuarcy is True. 
        In the contrary case it is nan.

    '''
    annotations = mne.read_annotations(filePath)
    annotations = markers_fixer(annotations)
    
    #Find right answers
    measurement = filePath[filePath.find('V'):filePath.find('V')+2]
    rightCircles = find_right_answers(measurement, 'SDR', rightAnswersPath) 
    
    # Some measurements had different identifiers (different versions) for the
    # target 
    target = 'large' if 'large' in annotations.description else 'largeCircle' 
    mask = np.logical_and(['Stimulus' in cue for cue in annotations.description],
                          [not 'S128' in cue for cue in annotations.description])
    
    askIdx = -1
    sdrScores = {}
    for i in range(annotations.__len__()):
        if 'ask' in annotations[i]['description']:
            print('\n ---------------- \n')
            print('Ask cue at ', annotations[i]['onset'], 's')
            askIdx += 1
            
            #Check there are more button cues after asking
            if sum(mask[i:]):
                rightPos = np.array(rightCircles[askIdx]) + 1
                print('Right postions were:', np.array(rightPos))
                nPos = len(rightPos+2) # the +2 is given to consider two more markers
                #into the evaluation as sometimes the patients pressed one button twice 
                #but they could have eventually pressed others that were right
                
                #Get idx of pressed button and compute time to query cue and
                #get the marker of the pressed button
                pressIdx = (i + np.argwhere(mask[i:])[0:nPos]).flatten().astype(int)
                
                #Validity check
                timeDiffs = np.array(annotations[pressIdx].onset) - annotations[i]['onset']
                print('Time difference to next button pressing: ', timeDiffs[0], 's')
                inTimePress = pressIdx[timeDiffs <= 5]
                print('The positions pressed in time were: ', annotations[inTimePress].description)
                
                #Check time and that there was no oddball target between the markers
                if any(inTimePress) and not target in annotations[i:inTimePress[-1]].description:
                    pressedPos = [int(marker.replace('Stimulus/S','').strip()) for 
                                  marker in annotations[inTimePress].description]
                    
                    # Swap to the ID convention of the source code (see Note)
                    if any(swapper != None):
                        #Double check there are no false or double button pressings
                        pressedPos = [pos for pos in pressedPos if pos in range(1,9)]
                        pressedPos = new_num_assigner(pressedPos, swapper)
                        print(pressedPos)
                    
                    # Count every right position 
                    # Note: rightPos have ALWAYS two different numbers
                    score = np.sum(np.isin(rightPos, pressedPos))
                    print(score, '/', nPos , ' positions correctly pressed.')
                    
                    #Score storage
                    if score:
                        sdrScores[askIdx] = (score, timeDiffs[0])
                    else:
                        sdrScores[askIdx] = (score, np.nan)
                        print('Wrong')
                else: 
                    sdrScores[askIdx] = (0, np.nan)
                    print('Invalid response')
            else:
                sdrScores[askIdx] = (0, np.nan)
                print('Pressed nothing afterwards')
    
    return sdrScores



def evaluate_oddball(filePath, rightAnswersPath=None, swapper = None):
    '''
    Evaluates the SDR task for a single measurement. It is a trial by trial 
    evaluation. It checks if anything was pressed after the oddball targets.
    For successful trials it computes the response time too.
    
    Information is stored in dictionary with information of all trials.

    Parameters
    ----------
    filePath : str
        path to .fif file with the measurement to be evaluated.
    rightAnswersPath : None
        passed, has no meaning in this function. It is put as parameter for 
        compatibility with the evaluate_SDR and evaluate_VDR function in the 
        evaluate_beh_patients function
    swapper : Numpy array or None.
        passed, has no meaning in this function. It is put as parameter for 
        compatibility with the evaluate_SDR function in the evaluate_beh_patients
        function
        
    Returns
    -------
    oddballScores : dict
        trial-by-trial scores obtained in the oddball task. Trial indexes
        are the keys for tuple of length 2. Position 0 has boolean value for 
        accuracy, position 1 has response time value if accuarcy is True. 
        In the contrary case it is nan..

    '''
    targets = ['large', 'largeCircle', 'highPitch']
    annotations = mne.read_annotations(filePath)
    
    #Start a score storing dictionary: key is index, 1st value is accuracy bool, 
    #2nd value is response time
    target = [i for i in annotations.description if i in targets][0]
    oddballScores = {}
    targetIdx = -1
    pressMask = ['Stimulus' in cue for cue in annotations.description]
    for i in range(annotations.__len__()):
        if annotations[i]['description'] == target:
            print('\n ---------------- \n')
            print('Target cue annotation #', i, 'at time ' , annotations[i]['onset'])
            targetIdx += 1
            #Check if there are press button markers after this one 
            if np.sum(pressMask[i:]):
                pressIdx = int(i + np.argwhere(pressMask[i:])[0][0])
                timeDiff = annotations[pressIdx]['onset'] - annotations[i]['onset']
                print('Time difference to next button pressing: ', timeDiff, 's')
                
                #Check if the time difference between the cue and pressing the button counts
                #Check if there was no ask cue in between 
                if timeDiff <= 3 and not 'askCircles' in annotations[i:pressIdx].description:
                    oddballScores[targetIdx] = (True, timeDiff)
                    print('Scored!')
                else:
                    oddballScores[targetIdx] = (False, np.nan)
                    print('Failed or invalid')
            else: 
                oddballScores[targetIdx] = (False, np.nan)
                print('Failed')
    return oddballScores


def compute_d_prime(hitRate, falseAlarmRate):
    '''
    Computes d prime or sensitivity index given hits and false positives rate

    Parameters
    ----------
    hitRate : float
    falseAlarmRate : float

    Returns
    -------
    dPrime : float

    '''
    return st.norm.ppf(hitRate) - st.norm.ppf(falseAlarmRate)


def oddball_sensitivity(filePath, rightAnswersPath=None, swapper=None):
    '''
    Computes the sensivity for a whole oddball measurement

    Parameters
    ----------
    filePath : str
        path to .fif file with the measurement to be evaluated.
    rightAnswersPath : None
        passed, has no meaning in this function. It is put as parameter for 
        compatibility with the evaluate_SDR and evaluate_VDR function in the 
        evaluate_beh_patients function
    swapper : Numpy array or None.
        passed, has no meaning in this function. It is put as parameter for 
        compatibility with the evaluate_SDR function in the evaluate_beh_patients
        function

    Returns
    -------
    dPrime : float
    falseNames : list of str
        list with the stimulus that were falsely identified as targets.

    '''
    
    annotations = mne.read_annotations(filePath)
    
    #Cross is a refernce but not counted as cue itself (only for visual oddball)
    crossMask = [marker != 'cross' for marker in annotations.description]
    annotations = annotations[crossMask]
    
    targets = ['large', 'largeCircle', 'highPitch']
    target = [i for i in annotations.description if i in targets][0]
    hits = 0
    falses = 0
    oddballNames = ['circle', 'distractor', 'large','highPitch','lowPitch',
                    'largeCircle']
    nonTargetNames = ['circle', 'distractor', 'lowPitch']
    falseNames = []
    
    for i in range(annotations.__len__()):
        name = annotations[i]['description']
        if name in oddballNames:
            if i < annotations.__len__() - 1:
                nextB = annotations[i+1]
                timeDiff = nextB['onset'] - annotations[i]['onset'] 
                if 'Stimulus' in nextB['description'] and timeDiff <= 3:
                    if name == target:
                        hits += 1
                    else:
                        falses += 1
                        falseNames.append(str(name))

    print('Hits:', hits, '.  False positives:', falses)
    totalTarget = np.sum([target == cue for cue in annotations.description])   
    totalNonTarget = np.sum([cue in nonTargetNames for cue in annotations.description])
    print('Total possible hits:', totalTarget, 'Total false positives:', totalNonTarget)
    
    #In case there are no false positives
    if not falses and hits:
        falses = 1
        totalNonTarget += 1
    
    # In case hits rate is 1 (perfect score)
    if int(hits) == int(totalTarget):
        totalTarget += 1
        
    dPrime = compute_d_prime(hits/totalTarget, falses/totalNonTarget)
    print('Sensitivity index:', dPrime)

    return (dPrime, falseNames)

def evaluate_beh_patients(filePathList, mode, rightAnswersPath=None, swapper=None):
    '''
    This function applies one of the behavioral evaluation functions to a set 
    of measurements from different patients. The evaluation to be performed 
    is defined by the parameter mode.

    Parameters
    ----------
    filePathList : list of str
        list with the paths to .fif file with the measurements to be evaluated.
    mode : str
        either 'SDR', 'VDR', 'oddball' or 'sensitivity'.
    rightAnswersPath : str, optional
        if mode is 'VDR' or 'SDR'. Path to .txt file with dictionary with the 
        right answers. The default is None.
    swapper : Numpy array, optional
        if mode is 'SDR' and button markers should be corrected. The default is None.

    Returns
    -------
    scoreAll : dict of dicts
        dict with subjects' IDs as keys. This dicts are further divided in dicts
        for each measurement. If of these is again a dictionary ouput of the
        evaluate_... functions

    '''
    taskDict = {'VDR':evaluate_VDR, 
                'SDR':evaluate_SDR, 
                'oddball':evaluate_oddball,
                'sensitivity': oddball_sensitivity}
    
    method = taskDict[mode]
    scoreAll = {}
    for m in filePathList:
        score = method(m, rightAnswersPath, swapper=swapper)
        name = Path(m).name # filename
        pat = name[:name.find('_')]
        meas = name[name.find('V'):name.find('_p_')]
        if pat not in scoreAll:
            scoreAll[pat] = {}    
        scoreAll[pat][meas] = score
    return scoreAll



def summarize_results(scores, sdr=False):
    '''
    summraizes the results of a measurement into total accuarcy ratio and mean
    response time

    Parameters
    ----------
    scores : dict
        scores dicitionary outputted by functions evaluate_SDR, evaluate_VDR,
        evaluate_oddball
    sdr :  bool, optional
        if the SDR task is being summarized.
    
    Returns
    -------
    total : float
        accuarcy ration scored in the measurement.
    respTime : float
        mean response time in the whole measurement.

    '''
    # In SDR the amount of possible right answers was 2 by trial
    total = len(scores)*2 if sdr else len(scores)
    points = np.sum([scores[s][0] for s in scores])/total
    respTime = np.nanmean([scores[s][1] for s in scores])
    
    print('RESULTS:')
    print(points*100, '% correct.')
    print('Average response time: ', respTime, 's' )
    return points, respTime


def summarize_all_results(scoresDict, sdr=False):
    '''
    Applies the summarizing results to sets of measurements and stores scores
    by patient and measurement in a dictionary

    Parameters
    ----------
    scoresDict : 
        dicitonary ouputted by the evaluate_beh_patients function
    sdr : bool, optional
        if the SDR task is being summarized.. The default is False.

    Returns
    -------
    sumResults : dict
        dict with all scores of a set of pateints' measurements summarized with
        scores by measurement and not trial.

    '''
    sumResults = {}
    for pat in scoresDict:
        if pat not in sumResults:
            sumResults[pat] = {}
        for meas in scoresDict[pat]:
            print('\n'*2, 'Patient:', pat, ' Measurement:', meas)
            sumResults[pat][meas] = summarize_results(scoresDict[pat][meas], sdr=sdr)
    return sumResults
