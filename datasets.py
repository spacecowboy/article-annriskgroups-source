# -*- coding: utf-8 -*-
'''
Load datasets with the relevant columns
'''
from __future__ import division, print_function
import pandas as pd
import numpy as np
import os

# Each data set defines: filename, timecol, eventcol, xcols

_pbc = dict(filename="data/pbc.csv",
            timecol="time",
            eventcol="status",
            xcols=["trt", "age", "sex", "ascites", "hepato", "spiders",
                   "edema", "bili", "chol", "albumin", "copper", "alk.phos",
                   "ast", "trig", "platelet", "protime", "stage"])

_lung = dict(filename="data/lung.csv",
             timecol="time",
             eventcol="status",
             xcols=["age", "sex", "ph.ecog", "ph.karno",
                    "pat.karno", "meal.cal", "wt.loss"])

_nwtco = dict(filename="data/nwtco.csv",
              timecol="edrel",
              eventcol="rel",
              xcols=["instit", "histol", "age", "stage"])

_flchain = dict(filename="data/flchain.csv",
                timecol="futime",
                eventcol="death",
                xcols=["age", "sex", "kappa", "lambda",
                       "flc.grp", "creatinine", "mgus"])

_colon = dict(filename="data/colon.csv",
              timecol="time",
              eventcol="status",
              xcols=["rx", "sex", "age", "obstruct", "perfor", "adhere",
                     "nodes", "differ", "extent", "surg", "node4"])


def split_dataframe_class_columns(df, cols, upper_lim=5, lower_lim=3, no_floats=True):
    '''
    Splits columns of a dataframe where rows can only take a limited
    amount of valid values, into seperate columns
    for each observed value. The result is a number of columns which are
    exclusive with each other: only one can be 1 at any time.

    Parameters:
    - df, pandas dataframe to work with
    - cols, eligible columns to split
    - upper_lim, only consider columns with less unique values (default 5)
    - lower_lim, only consider equal or more unique values (default 3)
    - no_floats, if True exclude columns containing decimal values

    Returns:
    A new pandas dataframe with the same columns as df, except those columns
    which have been split.

    Note: This function preserves NaNs.
    '''
    ndf = pd.DataFrame()
    for col in df.columns:
        uniques = df[col].unique()
        # Dont count nans as unique values
        nans = pd.isnull(uniques)
        uniques = uniques[~nans]
        # If class variable
        if (col in cols and
            (len(uniques) >= lower_lim and len(uniques) < upper_lim) and
            (not no_floats or np.all(uniques.astype(str) == uniques)
             or np.all(uniques.astype(int) == uniques))):
            # Split it, one col for each unique value
            for val in uniques:
                # A human-readable name
                ncol = "{}{}".format(col, val)
                # Set values
                ndf[ncol] = np.zeros_like(df[col])
                ndf.loc[df[col] == val, ncol] = 1
                # Also transfer NaNs
                ndf.loc[df[col].isnull(), ncol] = np.nan
        else:
            # Not a class variable
            ndf[col] = df[col]

    return ndf


def replace_dataframe_nans(df, binary_median=False):
    '''
    Replaces the NaNs of a pandas dataframe with
    the mean of the column, in case of continuous
    values. If the column is binary, it can be replaced
    with the median value if desired.

    Parameters:
    - df, the dataframe to replace NaNs in
    '''
    for col in df.columns:
        uniques = df[col].unique()
        # Dont count nans as unique values
        nans = pd.isnull(uniques)
        uniques = uniques[~nans]

        nans = pd.isnull(df[col])
        if binary_median and len(uniques) == 2:
            # Binary, use median
            df.loc[nans, col] = df[col].median()
        else:
            # Use mean
            df.loc[nans, col] = df[col].mean()


def normalize_dataframe(dataframe, cols=None, binvals=None):
    '''
    Normalize a pandas dataframe. Binary values are
    forced to (-1,1), and continuous (the rest) variables
    are forced to zero mean and standard deviation = 1

    Parameters:
    - dataframe, the pandas dataframe to normalize column-wise
    - cols, (optional iterable) the column names in the dataframe to normalize.
    - binvals, (default (0,1)) tuple giving the (min,max) binary values to use.

    Note: this function preserves NaNs.
    '''
    if cols is None:
        cols = dataframe.columns
    if binvals is None:
        binvals = (-1, 1)

    for col in cols:
        # Check if binary
        uniques = dataframe[col].unique()
        # Dont count nans as unique values
        nans = pd.isnull(uniques)
        uniques = uniques[~nans]

        if len(uniques) == 2:
            # Binary, force into 0 and 1
            mins = dataframe[col] == np.min(uniques)
            maxs = dataframe[col] == np.max(uniques)

            dataframe.loc[mins, col] = binvals[0]
            dataframe.loc[maxs, col] = binvals[1]
        else:
            # Can still be "binary"
            if len(uniques) == 1 and uniques[0] in [0, 1]:
                # Yes, single binary value
                continue

            # Continuous, zero mean with 1 standard deviation
            mean = dataframe[col].mean()
            std = dataframe[col].std()

            dataframe.loc[:, col] = dataframe[col] - mean
            # Can be single value
            if std > 0:
                dataframe.loc[:, col] = dataframe[col] / std


def get_data(filename, timecol, eventcol, xcols, norm_in=True, norm_out=True,
             prints=False, splitcols=None, training=True, setcol='set'):
    '''
    Parse the data.

    Returns a DataFrame where the first two columns are always (time, event).
    '''
    if not isinstance(filename, list):
        filename = [filename]

    # Suppot concatenating several files
    _d = None
    for fname in filename:
        _dtemp = pd.read_csv(fname, sep=None, engine='python')

        if 'pbc' in fname:
            # For pbc/mayo, only use first 312 entries
            _dtemp = _dtemp.reindex(np.arange(312))
            # Change status to binary
            _dtemp.loc[_dtemp[eventcol] < 2, eventcol] = 0
            _dtemp.loc[_dtemp[eventcol] == 2, eventcol] = 1
            # Sex to numbers
            _dtemp.loc[_dtemp['sex'] == 'f', 'sex'] = 0
            _dtemp.loc[_dtemp['sex'] == 'm', 'sex'] = 1
        elif 'lung' in fname:
            # Status is listed as 1=cens, 2=death
            _dtemp.loc[_dtemp[eventcol] < 2, eventcol] = 0
            _dtemp.loc[_dtemp[eventcol] == 2, eventcol] = 1
        elif 'flchain' in fname:
            # Sex variable to numbers
            _dtemp.loc[_dtemp['sex'] == 'F', 'sex'] = 0
            _dtemp.loc[_dtemp['sex'] == 'M', 'sex'] = 1
        elif 'colon' in fname:
            # Filter on recurrence, ignore death entries
            _dtemp = _dtemp.reindex(_dtemp.loc[_dtemp['etype'] == 1].index)

        # Will use this after normalization
        if training:
            which = (_dtemp[setcol] == 'training')
        else:
            which = (_dtemp[setcol] == 'testing')

        # Make sure order is correct and other columns are dropped
        _dtemp = _dtemp.reindex(columns=([timecol, eventcol] + list(xcols)))

        if _d is None:
            _d = _dtemp
        else:
            _d = pd.concat([_d, _dtemp], ignore_index=True)

    inshape = _d.shape

    # Split columns into binary
    if splitcols is None:
        # Split all appropriate ones
        _d = split_dataframe_class_columns(_d, cols=_d.columns)
    elif len(splitcols) > 0:
        _d = split_dataframe_class_columns(_d, cols=splitcols)

    # Rename columns with parenthesis in them - R doesn't like that
    c = list(_d.columns)
    for i, name in enumerate(c):
        if '(' in name or ')' in name or '=' in name or '+' in name:
            c[i] = name.replace('(', '').replace(')', '').replace('=', '').replace('+', '')
    _d.columns = c

    # Include new columns
    xcols = list(_d.columns)
    xcols.remove(timecol)
    xcols.remove(eventcol)

    # Normalize input columns, except time and events
    if norm_in:
        normalize_dataframe(_d, cols=xcols, binvals=(-1,1))
    if norm_out:
        normalize_dataframe(_d, cols=[timecol])

    # Fill missing values with mean/median
    replace_dataframe_nans(_d, binary_median=False)

    # Return only training/testing subset
    _d = _d.reindex(_d.loc[which].index)

    E = _d[eventcol]
    if prints:
        outshape = _d.shape
        if norm_in:
            print("Input columns were normalized")
        if norm_out:
            print("Target column was normalized")
        print("Shape from {} to {}".format(inshape, outshape))
        print("Censored count:", len(_d) - np.sum(E), "/", len(_d),
              " = {:.2f}%".format(100*(len(_d) - np.sum(E))/len(_d)))
        print("Final columns:", list(_d.columns))
    return _d.astype(float)


def get_pbc(prints=False, norm_in=False, norm_out=False, training=True):
    # Do not split columns in this dataset
    return get_data(prints=prints, norm_in=norm_in, norm_out=norm_out,
                    splitcols=[], training=training, **_pbc)


def get_lung(prints=False, norm_in=False, norm_out=False, training=True):
    # Do not split columns in this dataset
    return get_data(prints=prints, norm_in=norm_in, norm_out=norm_out,
                    splitcols=[], training=training, **_lung)


def get_nwtco(prints=False, norm_in=False, norm_out=False, training=True):
    return get_data(prints=prints, norm_in=norm_in, norm_out=norm_out,
                    training=training, **_nwtco)


def get_colon(prints=False, norm_in=False, norm_out=False, training=True):
    # Only split rx
    return get_data(prints=prints, norm_in=norm_in, norm_out=norm_out,
                    splitcols=['rx'], training=training, **_colon)


def get_flchain(prints=False, norm_in=False, norm_out=False, training=True):
    # Only split rx
    return get_data(prints=prints, norm_in=norm_in, norm_out=norm_out,
                    splitcols=None, training=training, **_flchain)
