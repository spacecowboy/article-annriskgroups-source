
# coding: utf-8

# In[1]:

# import stuffs
import numpy as np
import pandas as pd

from pickle import dump
import os
import sys
import random

from pysurvival.rpart import RPartModel
from classcox import CoxClasser

import ann
from classensemble import ClassEnsemble
from helpers import get_net

from datasets import get_colon, get_nwtco, get_flchain, get_pbc, get_lung


if len(sys.argv) != 2:
    exit("Usage: script.py NUMERIC_ID")

run_id = int(sys.argv[1])

if run_id < 1:
    exit("Please give a positive number!")


# Training data
datasets = {}

# Add the data sets
for name, getter in zip(["pbc", "lung", "colon", "nwtco", "flchain"],
                        [get_pbc, get_lung, get_colon, get_nwtco, get_flchain]):
    trn = getter(norm_in=True, norm_out=False, training=True)
    datasets[name] = trn

    cens = (trn.iloc[:, 1] == 0)
    censcount = np.sum(cens) / trn.shape[0]
    print(name, "censed:", censcount)


# Crossval variables
cross_n = 1
cross_k = 3


# In[3]:

# Save predictions for later
rpart_val_preds = {}
for dname in datasets.keys():
    # N lists : Each k items later
    rpart_val_preds[dname] = [[] for _ in range(cross_n)]


# Default values in rpart actually
rpart_kwargs = dict(highlim=0.15,
                    lowlim=0.15,
                    minsplit=20,
                    minbucket=None,
                    xval=3,
                    cp=0.01)

# Networks depend on rparts training values
rpart = RPartModel(**rpart_kwargs)


# In[4]:


# Save predictions for later
cox_val_preds = {}
for dname in datasets.keys():
    # N lists : Each k items later
    cox_val_preds[dname] = [[] for _ in range(cross_n)]


# In[ ]:

print("Doing ANN...")

# Save predictions for later
ann_val_preds = {}
for dname in datasets.keys():
    # N lists : Each k items later
    ann_val_preds[dname] = [[] for _ in range(cross_n)]


def get_ensemble(incols, high_size, low_size):
    hnets = []
    lnets = []

    netcount = 34
    for i in range(netcount):
        if i % 2:
            n = get_net(incols, high_size, ann.geneticnetwork.FITNESS_SURV_KAPLAN_MIN)
            hnets.append(n)
        else:
            n = get_net(incols, low_size, ann.geneticnetwork.FITNESS_SURV_KAPLAN_MAX)
            lnets.append(n)

    return ClassEnsemble(hnets, lnets)


# In[ ]:

# Save each random permutation
data_permutations = {}
for dname in datasets.keys():
    # N lists : Each k items later
    data_permutations[dname] = [[] for _ in range(cross_n)]


# Repeat cross validation
for rep in range(cross_n):
    print("n =", rep)
    # For each data set
    for dname, _df in datasets.items():
        print("Dataset:", dname)
        n, d = _df.shape
        k = cross_k

        duration_col = _df.columns[0]
        event_col = _df.columns[1]
        testing_columns = _df.columns - [duration_col, event_col]

        # Random divisions, stratified on events
        perm = np.random.permutation(_df.index)
        # Save it for later
        data_permutations[dname][rep].append(perm)

        df = _df.reindex(perm).sort(event_col)

        assignments = np.array((n // k + 1) * list(range(1, k + 1)))
        assignments = assignments[:n]

        # For each division
        for i in range(1, k + 1):
            ix = assignments == i
            training_data = df.ix[~ix]
            testing_data = df.ix[ix]

            #T_actual = testing_data[duration_col].values
            #E_actual = testing_data[event_col].values
            #X_testing = testing_data[testing_columns]

            # Train rpart first
            rpart.fit(training_data, duration_col, event_col)

            rpart_val_preds[dname][rep].append(rpart.predict_classes(testing_data))

            total = training_data.shape[0]
            high_size = rpart.high_size
            low_size = rpart.low_size

            # Cox uses quartile formulation 0 - 100
            cox = CoxClasser(100 * (1 - high_size / total),
                     100 * low_size / total)
            cox.fit(training_data, duration_col, event_col)
            cox_val_preds[dname][rep].append(cox.predict_classes(testing_data))

            # ANN
            net = get_ensemble(len(testing_columns), high_size, low_size)
            net.fit(training_data, duration_col, event_col)
            ann_val_preds[dname][rep].append(net.predict_classes(testing_data))



# In[ ]:


# These are expensive, so save to disk
crossval_results = {}
crossval_results['data_permutations'] = data_permutations
crossval_results['rpart_val_preds'] = rpart_val_preds
crossval_results['cox_val_preds'] = cox_val_preds
crossval_results['ann_val_preds'] = ann_val_preds

path = "crossval-{}.pickle".format(run_id)
while os.path.exists(path):
    print("File exists. Should not be overwritten")
    s = list('0123456789ABCDEF')
    rand_id = ''.join([s[random.randint(0, len(s) - 1)] for _ in range(6)])
    path = "crossval-{}-{}.pickle".format(run_id, rand_id)

with open(path, 'wb') as F:
    dump(crossval_results, F)
