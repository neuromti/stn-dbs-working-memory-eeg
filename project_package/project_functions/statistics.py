# -*- coding: utf-8 -*-
"""Statistics and plotting helpers for analysis outputs."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns 


#%% Functions
def add_norm_values(data, columns=[]):
    '''
    Adds normalized values to a data frame given the list of columns from 
    which the normalized values should be added. Columns with the same names 
    + 'Norm' are added.

    Parameters
    ----------
    data : pandas DataFrame
    columns : list of str, optional
        List with the names of the columns, from which the values should be 
        normalized and added as new columns. The values in this columns need
        to be numerical. The default is [].

    Returns
    -------
    data : pandas DataFrame
        the input data frame with the added columns containing the normalized 
        values.

    '''
    for col in columns:
        for i in data.index:
            baseline = data[(data['ID'] == data.iloc[i]['ID']) & 
                            (data['time'] == 'V0') &
                            (data['task'] == data.iloc[i]['task'])][col]
            if baseline.__len__():
                baseline = baseline.item()
                value = data.loc[(i, col)].item()
                data.loc[(i, col+'Norm')] = (value - baseline)/baseline
    return data

def add_baseline_values(data, columns=[]):
    '''
    Adds baseline values to a data frame given the list of columns from 
    which the normalized values should be added. Columns with the same names 
    + 'BL' are added.
    
    It facilitates correlation computations

    Parameters
    ----------
    data : pandas DataFrame
    columns : list of str, optional
        List with the names of the columns, from which the baseline values should 
        be added as new columns. The default is [].

    Returns
    -------
    data : pandas DataFrame
        the input data frame with the added columns containing the baseline 
        values.
    '''
    for col in columns:
        for i in data.index:
            baseline = data[(data['ID'] == data.iloc[i]['ID']) & 
                            (data['time'] == 'V0') &
                            (data['task'] == data.iloc[i]['task'])][col]
            if len(baseline):
                data.loc[(i,col+'BL')] = baseline.item()
    return data


def compute_correlation(data, drop_columns=[], plot=False): 
    '''
    Computes correlation of series of data in a data frame and the corresponding
    p-values

    Parameters
    ----------
    data : pandas DataFrame
    drop_columns : list of str, optional
        columns that should not be taken into account in the computations.
        Columns that have non-numberical values must be lister here. If not, 
        the function will fail. The default is [].
    plot : bool, optional
        whether the correltion matrix should be plotted as a heatmap. 
        The default is False.

    Returns
    -------
    corr : DataFrame
        correlation matrix of the input dataset except from columns in 'drop_columns'.
        The entries are the r Pearson coefficients
    pval : DataFrame
        p-values corresponding to the correlations in corr.
    sign : DataFrame
        data frame with boolean values on non-significance of the r coefficients
        in corr. Useful for plotting a masked correlation matrix

    '''
    corr = pd.DataFrame()
    sign = pd.DataFrame()
    pval = pd.DataFrame()

    for col1 in data.columns.drop(drop_columns):
        for col2 in data.columns.drop(drop_columns):
            r, p = st.pearsonr(data[col1], data[col2])
            corr.loc[(col1, col2)] = r
            pval.loc[(col1, col2)] = p
            sign.loc[(col1, col2)] = (p > 0.05)
    
    if plot:
        fig, ax = plt.subplots()
        sns.heatmap(corr, cmap='RdBu_r', mask=sign, vmin=-1, annot_kws={'size':8}) 
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                       rotation_mode="anchor", size=8)
        plt.setp(ax.get_yticklabels(), size=8)
        
        
    return corr, pval, sign

def corr_df(data, colnames=[], rownames=[], drop_columns=[]):
    '''
    Computes correlations of given variables in a data frame. It applies the 
    compute_correlation function and the organizes the variables subset specified
    with the relevant statistical values. Variables in the
    columns and rows can be specified separately. The r coefficients and p-values
    are given for each computation

    Parameters
    ----------
    data : pandas DataFrame
        data frame with the name of variables as columns
    colnames : list of str, optional
        List of variables (need to be columns of the given dataframe) to be used
        as columns in the correlation data frame. The default is [].
    rownames : list of str, optional
        List of variables (need to be columns of the given dataframe) to be used
        as columns in the correlation data frame. The default is [].
    drop_columns : list of str, optional
        columns that should not be taken into account in the computations.
        Applies for the whole data frame!
        Columns that have non-numberical values must be lister here. If not, 
        the function will fail. The default is [].
        
    Returns
    -------
    corrDf : pandas DataFrame
        data frame with correlation measures between the variables in colnames
        and rownames.

    '''
    corr, pval, _ = compute_correlation(data, drop_columns=drop_columns)
    corrDf = pd.DataFrame()
    for col in colnames:
        for row in rownames:
            corrDf.loc[(row, col+'RCoeff')] = corr.loc[row][col]
            corrDf.loc[(row, col+'PValue')] = pval.loc[row][col]
    return corrDf

def subsetter(data, task=None, time=None, analizableDf=None):
    '''
    Generates subset of data given the data frame and the information of analizable
    measurements with each other. 

    Parameters
    ----------
    data : pandas DataFrame
        it must have a column names 'ID' referring to subjects' IDs and a column
        names time with the identifiers for the measruements in the convention
        of the project.
    task : 'SDR', 'VDR' or None, optional
        The default is None. If None there is no selection of measurements of
        a certain task. 
    time : 'short', 'long' or None, optional
        The default is None. If None there is no selection of measurements of
        a certain time subset. 
    analizableDf : pandas DataFrame, optional
        It contains boolean values for each patient, task and time scale on 
        its suitability to be included in the subset. 
        It should be always included to ensure the validity of the analysis.
        The default is None.

    Returns
    -------
    data : pandas DataFrame
        subset of the given dataframe containing only the measurments (rows)
        suited to be analized in the task and timescale given.

    '''
    # Generate subset by task and time 
    if task:
        data = data[data['task'] == task]
    if (time == 'short') or (time == 'long'):
        timeDict = {'short': ['V3_1', 'V3_2', 'V3_3'],
                    'long': ['V0', 'V4', 'V5']}
        data = data[np.isin(data['time'], timeDict[time])]
        
    # Keep only informtion of subjects suited for the analysis
    if any(analizableDf != None):
        idx = analizableDf[analizableDf[task+'_'+time]]['ID']
        data = data[np.isin(data['ID'], idx)]
    return data
        
def significance_df(data):
    '''
    It converts data frames with statistical values adding the conventions for 
    significance on the p-values (* and **)
    
    It is useful fot the report of the results
    
    Parameters
    ----------
    data : DataFrame
        data frame with statistical values/coefficients, etc. The columns with 
        p -values must have 'PValue' in the name

    Returns
    -------
    sdata : DataFrame
        Same input data frame but with significance conventions and rounded values. 
        P-values lower than 0.001 are turned into '< 0.001 **'. This frame is only 
        for documentation purposes as all values are converted to strings

    '''
    sdata = pd.DataFrame()

    for col in data.columns:
        for i in data.index:
            value = data.loc[(i, col)]
            #Pass only columns with numerical values 
            if type(value) != str:
               
                if 'PValue' in col: 
                    if value > 0.05:
                        sdata.loc[(i, col)] = str(round(value, 3))
                    elif value > 0.001:
                        sdata.loc[(i, col)] = str(round(value, 3)) + '*'
                    else:
                        sdata.loc[(i, col)] =  '< 0.001 **'
                #round also values that are not p-values        
                else:
                    sdata.loc[(i, col)] = str(round(value, 3))
            # Keep all other info of the df
            else:
                sdata.loc[(i, col)] = value
    return sdata

def renamer(var, task):
    '''
    Given the variables used during the analysis in Python generate informative
    label names for axis of figures

    Parameters
    ----------
    var : str
        name of variable.
    task : 'VDR' or 'SDR'
        measurement type. For oddball variables the dleayed response task 
        performed in parallel

    Returns
    -------
    new : str
        new string suited for labeling axes. It contains unit info

    '''
    dc = {'SDR': 'visual oddball',
          'VDR': 'auditory oddball'}
    if 'task' in var:
        new = task + ' task '
        if 'Acc' in var:
            new = new + ' accuracy'
        elif 'RT' in var:
            new = new + ' RT'
    elif 'oddball' in var:
        new = dc[task]
        if 'Acc' in var:
            new = new + ' accuarcy'
        elif 'RT' in var:
            new = new + ' RT'
    elif 'sensitivity' in var:
        new = dc[task] +' sensitivity index'
    elif 'P3' in var:
        unit = {'amplitude': ' [mV/$m^2$]',
                'latency': ' [s]'}
        stm = [s for s in ['distractor','nontarget', 'target'] if s in var][0]
        if stm == 'nontarget':
            stm = 'non-target'
        ftr = [f for f in ['amplitude', 'latency', 'width'] if f in var][0]
        new = stm+'-related P3 '+ ftr + unit[ftr]+' ('+ dc[task] + ')'
    elif 'TF' in var:
        bnd = [b for b in ['alpha', 'beta', 'theta', 'delta', 'total'] if b in var][0]
        new = bnd+' power [dB]'
    else:
        new= var
    if 'Norm' in var:
        new = new + ' (normalized)'
    if 'BL' in var:
        new = new + ' (baseline)'
    return new

def plot_for_stats(data, depvar, indvar, task=None, time=None, analizableDf=None,
                   hue=None):
    '''
    Plots boxplots with stripplots for statstics.

    Parameters
    ----------
    data : pandas DataFrame
    depvar : str
        name of dependent variable. The data point to be displayed along the y 
        axis.
    indvar : str
        name of independent variable. The categories along the x-axis
    task : 'VDR' or 'SDR', optional
        task subset to be considered. For oddball variables the dleayed response task 
        performed in parallel The default is None.
    time : str, optional
        time subset to be considered. The default is None.
    analizableDf : DataFrame, optional
        It contains boolean values for each patient, task and time scale on 
        its suitability to be included in the subset. 
        It should be always included to ensure the validity of the analysis.
        The default is None.
    hue : str, optional
        subcategories in which to further divide the data. If used, the 
        stripplots are removed.The default is None.

    Returns
    -------
    None.

    '''
        
    data = subsetter(data, task=task, time=time, analizableDf=analizableDf)
    
    if time == 'long':
        data = data[data['time'] != 'V0']
    
    colors=['deepskyblue','limegreen' ,'aquamarine', 'yellowgreen', 'mediumblue']
    fill_palette=sns.color_palette(colors)
    sns.set_palette(fill_palette)  
    
    order = ['OFF', 'DIR', 'OMNI'] if (indvar == 'stimulation') and (time != 'long') else None
    hue_order = ['OFF', 'DIR', 'OMNI'] if (hue == 'stimulation') \
        and (time != 'long') and (hue != None) else None
    
    plt.figure()
    ax = sns.boxplot(x=indvar, y=depvar, data=data,hue =hue, orient="v", 
                     fliersize=3, saturation=0.4, linewidth=1, order = order, 
                     hue_order = hue_order, whis=1.5)
    if hue == None:
        sns.stripplot(x=indvar, y=depvar, data=data, color='.3', order = order)
    else:
        plt.legend(loc = 'lower left')
    ax.set_ylabel(renamer(depvar, task), fontdict={'fontfamily':'sans-serif',
                                             'fontsize':10,
                                             'fontweight': 'book',
                                             'variant': 'small-caps'})
    
    
    sns.despine()
    
def visualize_corr_matrix(data):
    '''
    Visualization of correlation in data frame. Not masked.

    Parameters
    ----------
    data : pandas DataFrame

    Returns
    -------
    None.

    '''
    fig, ax = plt.subplots()
    plt.imshow(data, cmap = 'RdBu', vmin=-1)
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_yticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns, size=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", size=8)
    
    plt.colorbar()
