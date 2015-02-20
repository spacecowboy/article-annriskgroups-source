# -*- coding: utf-8 -*-
'''
Load datasets with the relevant columns
'''
from __future__ import division, print_function
import pandas as pd
import numpy as np
import os

# datasets to work on
# To handle each data set I need: filename, timecol, eventcol, xcols
# This is thus a list of tuples: (name, dict)
datasets = []

_breasttrnpath = "~/DataSets/breast_cancer_1/n4369_utanmalmo_trainingtwothirds.csv"
_breasttrn = dict(filename=os.path.expanduser(_breasttrnpath),
                  timecol='time_10y',
                  eventcol='event_10y',
                  xcols=['age', 'lymfmet', 'log(1+lymfmet)', 'n_pos',
                         'tumsize', 'log(1+er_cyt)', 'log(1+pgr_cyt)',
                         'pgr_cyt_pos', 'er_cyt_pos', 'size_gt_20',
                         'er_cyt', 'pgr_cyt'])
datasets.append(("breast_trn", _breasttrn))


_breasttestpath = "~/DataSets/breast_cancer_1/n4369_utanmalmo_targetthird.csv"
_breasttest = dict(filename=os.path.expanduser(_breasttestpath),
                   timecol='time_10y',
                   eventcol='event_10y',
                   xcols=['age', 'lymfmet', 'log(1+lymfmet)', 'n_pos',
                          'tumsize', 'log(1+er_cyt)', 'log(1+pgr_cyt)',
                          'pgr_cyt_pos', 'er_cyt_pos', 'size_gt_20',
                          'er_cyt', 'pgr_cyt'])
datasets.append(("breast_test", _breasttest))


_mayopath = "~/DataSets/vanBelle2009/Therneau2000/data_with_event_only_randomized.csv"
_mayo = dict(filename=os.path.expanduser(_mayopath),
             timecol='time',
             eventcol='event',
             xcols="trt,age,sex(m=1),ascites,hepato,spiders,edema,bili,chol".split(","))
datasets.append(("mayo", _mayo))


_lungpath = "~/DataSets/vanBelle2009/Therneau2000-MLC/data-full.csv"
_lung = dict(filename=os.path.expanduser(_lungpath),
             timecol="time",
             eventcol="status",
             xcols="age,sex,ph.ecog,ph.karno,pat.karno,meal.cal,wt.loss".split(","))
datasets.append(("lung", _lung))


_veteranpath = "~/DataSets/vanBelle2009/Kalbfleish2002/data.csv"
_veteran = dict(filename=os.path.expanduser(_veteranpath),
                timecol="time",
                eventcol="event",
                xcols="treatment,celltype,karnofsky,diagmonths,age,prior".split(","))
datasets.append(("veteran", _veteran))


_transplantpath1 = "~/DataSets/heart_transplant/final_normalized_develop.csv"
_transplantpath2 = "~/DataSets/heart_transplant/final_normalized_interntest.csv"
_transplant = dict(filename=[os.path.expanduser(_transplantpath1),
                             os.path.expanduser(_transplantpath2)],
                   timecol="survdays",
                   eventcol="nonsurvivor",
                       xcols="donage;recageyear;donsex;recsex;recdonheightmatch;recdonweightmatch;recpvr;recweightkg;recheighthcm;recobstpulmdisease;recsmokehist;rechypertension;recvasculardisease;reccvi;recstroke;recdiabetes;recinfections2weeks;recpepticulcer;recdialysis;recunstableangina;recdefibrillator;recantiarrhythmics;recamiodarone;reccreatininemostrecent;recbilirubin;recalbumin;recpasys;recinotropsup;recventilator;recvad;rececmo;reciabp;recpriortx;recpriorcardsurg;reccmvpreop;recmalignpretx;recworkincometx;recexercise_o2;recprevtxfus;recpra10;mmhla_a;mmhla_b;mmhla_dr;ischemictimeminutes;doncmvresult;donhypertension;dondiabetes;donlvef;donbilirubin;doncreatinine;donbun;donsmokehist;donalcoholhist;doncocainehist;donweight;donheight;diagn_CAD;diagn_Cardiomyopathy;diagn_Congenital;diagn_Graftfailure;diagn_Misc;diagn_Valve;donbldgroup_A;donbldgroup_AB;donbldgroup_B;donbldgroup_O;dondeath_Head trauma;dondeath_Other;dondeath_Stroke;recbldgroup_A;recbldgroup_AB;recbldgroup_B;recbldgroup_O;recmedcond_Home;recmedcond_Hospital;recmedcond_ICU;txyear_1991-1995;txyear_1996-2000;txyear_2001-2005;txyear_2006-2010".split(";"))
datasets.append(("transplant", _transplant))


def split_dataframe_class_columns(df, upper_lim=5, lower_lim=3, int_only=True):
    '''
    Splits columns of a dataframe where rows can only take a limited
    amount of valid values, into seperate columns
    for each observed value. The result is a number of columns which are
    exclusive with each other: only one can be 1 at any time.

    Parameters:
    - df, pandas dataframe to work with
    - upper_lim, only consider columns with less unique values (default 5)
    - lower_lim, only consider equal or more unique values (default 3)
    - int_only, if True only include columns with all integers

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
        if ((len(uniques) >= lower_lim and len(uniques) < upper_lim) and
            (not int_only or np.all(uniques.astype(int) == uniques))):
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
             prints=False, splitcols=None):
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

        # Make sure order is correct and other columns are dropped
        _dtemp = _dtemp.reindex(columns=([timecol, eventcol] + list(xcols)))

        if _d is None:
            _d = _dtemp
        else:
            _d = pd.concat([_d, _dtemp], ignore_index=True)

    inshape = _d.shape

    # Split columns into binary
    # Hard code some data set things
    if 'ph.ecog' in xcols:
        # Don't split lung data set
        pass
    elif splitcols is None:
        # Split all appropriate ones
        _d = split_dataframe_class_columns(_d)
    elif len(splitcols) > 0:
        _d[splitcols] = split_dataframe_class_columns(_d[splitcols])

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
    return _d


def get_mayo(prints=False, norm_in=False, norm_out=False):
    return get_data(prints=prints, norm_in=norm_in, norm_out=norm_out,
                    **_mayo)


def get_lung(prints=False, norm_in=False, norm_out=False):
    return get_data(prints=prints, norm_in=norm_in, norm_out=norm_out,
                    **_lung)


def get_veteran(prints=False, norm_in=False, norm_out=False):
    return get_data(prints=prints, norm_in=norm_in, norm_out=norm_out,
                    **_veteran)


def get_transplant(prints=False, norm_in=False, norm_out=False):
    return get_data(prints=prints, norm_in=norm_in, norm_out=norm_out,
                    **_transplant)


def get_breasttrn(prints=False, norm_in=False, norm_out=False):
    return get_data(prints=prints, norm_in=norm_in, norm_out=norm_out,
                    **_breasttrn)


def get_breasttest(prints=False, norm_in=False, norm_out=False):
    return get_data(prints=prints, norm_in=norm_in, norm_out=norm_out,
                    **_breasttest)
