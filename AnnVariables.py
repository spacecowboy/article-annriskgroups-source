
# coding: utf-8

# In[1]:

# import stuffs
#get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
from pyplotthemes import get_savefig, classictheme as plt
plt.latex = True


# In[2]:

from datasets import get_mayo, get_veteran, get_lung, get_breasttrn

d = get_mayo(prints=True, norm_in=True, norm_out=False)

durcol = d.columns[0]
eventcol = d.columns[1]

if np.any(d[durcol] < 0):
    raise ValueError("Negative times encountered")

print("End time:", d.iloc[:, 0].max())
d


# In[3]:

import ann
from classensemble import ClassEnsemble


def get_net(rows, incols, func=ann.geneticnetwork.FITNESS_SURV_KAPLAN_MIN, mingroup=None,
            hidden_count=3, popsize=100, generations=200, mutchance=0.15, conchance=0,
            crossover=ann.geneticnetwork.CROSSOVER_UNIFORM,
            selection=ann.geneticnetwork.SELECTION_TOURNAMENT):
    outcount = 2
    l = incols + hidden_count + outcount + 1

    net = ann.geneticnetwork(incols, hidden_count, outcount)
    net.fitness_function = func
    
    if mingroup is None:
        mingroup = int(0.25 * rows)
    
    # Be explicit here even though I changed the defaults
    net.connection_mutation_chance = conchance
    net.activation_mutation_chance = 0
    # Some other values
    net.crossover_method = crossover
    net.selection_method = selection
    net.population_size = popsize
    net.generations = generations
    net.weight_mutation_chance = mutchance


    ann.utils.connect_feedforward(net, hidden_act=net.TANH, out_act=net.SOFTMAX)
    c = net.connections.reshape((l, l))
    c[-outcount:, :(incols + hidden_count)] = 1
    net.connections = c.ravel()
    
    return net

def _netgen(df, netcount, funcs=None, **kwargs):
    # Expects (function, mingroup)
    if funcs is None:
        funcs = [ann.geneticnetwork.FITNESS_SURV_KAPLAN_MIN,
                 ann.geneticnetwork.FITNESS_SURV_KAPLAN_MAX]
        
    rows = df.shape[0]
    incols = df.shape[1] - 2
    hnets = []
    lnets = []

    for i in range(netcount):
        if i % 2:
            n = get_net(rows, incols, funcs[0], **kwargs)
            hnets.append(n)
        else:
            n = get_net(rows, incols, funcs[1], **kwargs)
            lnets.append(n)
    
    return hnets, lnets

def _kanngen(df, netcount, **kwargs):
    return _netgen(df, netcount, **kwargs)

def _riskgen(df, netcount, **kwargs):
    return _netgen(df, netcount,
                   [ann.geneticnetwork.FITNESS_SURV_RISKGROUP_HIGH,
                    ann.geneticnetwork.FITNESS_SURV_RISKGROUP_LOW],
                   **kwargs)

def get_kanngen(netcount, **kwargs):
    return lambda df: _kanngen(df, netcount, **kwargs)
#e = ClassEnsemble(netgen=netgen)
#er = ClassEnsemble(netgen=riskgen)

class NetFitter(object):
    def __init__(self, func=ann.geneticnetwork.FITNESS_SURV_KAPLAN_MIN, **kwargs):
        self.kwargs = kwargs
        self.func = func
        
    def fit(self, df, duration_col, event_col):
        '''
        Same as learn, but instead conforms to the interface defined by
        Lifelines and accepts a data frame as the data. Also generates
        new networks using self.netgen is it was defined.
        '''
        rows = df.shape[0]
        incols = df.shape[1] - 2
        self.net = get_net(rows, incols, self.func, **self.kwargs)
        # Save columns for prediction later
        self.x_cols = df.columns - [duration_col, event_col]
        self.net.learn(df[self.x_cols].values,
                       df[[duration_col, event_col]].values)
        
    def get_log(self, df):
        '''
        Returns a truncated training log
        '''
        return pd.Series(self.net.log.ravel()[:self.net.generations])


# In[4]:

from stats import k_fold_cross_validation
from lifelines.estimation import KaplanMeierFitter, median_survival_times


def score(T_actual, labels, E_actual):
    '''
    Return a score based on grouping
    '''
    scores = []
    labels = labels.ravel()
    for g in ['high', 'mid', 'low']:
        members = labels == g
        
        if np.sum(members) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(T_actual[members],
                    E_actual[members],
                    label='{}'.format(g))
            
            # Last survival time
            if np.sum(E_actual[members]) > 0:
                lasttime = np.max(T_actual[members][E_actual[members] == 1])
            else:
                lasttime = np.nan
        
            # End survival rate, median survival time, member count, last event
            subscore = (kmf.survival_function_.iloc[-1, 0],
                        median_survival_times(kmf.survival_function_),
                        np.sum(members),
                        lasttime)
        else:
            # Rpart might fail in this respect
            subscore = (np.nan, np.nan, np.sum(members), np.nan)
            
        scores.append(subscore)
    return scores

def logscore(T_actual, log, E_actual):
    # Return last value in the log
    return log[-1]


# # Compare stuff

# In[ ]:

#netcount = 6
models = []
# Try different epoch counts
for x in [0.0, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5]:
    #e = ClassEnsemble(netgen=get_kanngen(netcount, generations=x))
    e = NetFitter(func=ann.geneticnetwork.FITNESS_SURV_KAPLAN_MIN, 
                  popsize=50, generations=100, conchance=x)
    #, mingroup=int(0.25*d.shape[0]))
    e.var_label = 'Connection chance'
    e.var_value = x
    models.append(e)


# In[ ]:

n = 10
k = 4
# Repeated cross-validation
repeat_results = []
for rep in range(n):
    result = k_fold_cross_validation(models, d, durcol, eventcol, k=k, evaluation_measure=logscore, predictor='get_log')
    repeat_results.append(result)
#repeat_results


# # Plot results

# In[ ]:

def plot_logscore(repeat_results, models):
    boxes = []
    labels = []
    var_label = None
    # Makes no sense for low here for many datasets...
    for i, m in enumerate(models):
        labels.append(str(m.var_value))
        var_label = m.var_label
        vals = []
        for result in repeat_results:
            vals.extend(result[i])
        boxes.append(vals)

    plt.figure()
    plt.boxplot(boxes, labels=labels, vert=False, colors=plt.colors[:len(models)])
    plt.ylabel(var_label)
    plt.title("Cross-validation: n={} k={}".format(n, k))
    plt.xlabel("Something..")
    #plt.gca().set_xscale('log')
        
plot_logscore(repeat_results, models)


# In[ ]:

def plot_score(repeat_results, models, scoreindex=0):
    boxes = []
    labels = []
    var_label = []
    # Makes no sense for low here for many datasets...
    for i, g in enumerate(['high', 'mid', 'low']):
        for j, m in enumerate(models):
            if g == 'high':
                labels.append('H ' + str(m.var_value))
            elif g == 'low':
                labels.append('L ' + str(m.var_value))
            else:
                labels.append('M ' + str(m.var_value))
            var_label = m.var_label
            vals = []
            for result in repeat_results:
                for subscore in result[j]:
                    vals.append(subscore[i][scoreindex])
            boxes.append(vals)

    plt.figure()
    plt.boxplot(boxes, labels=labels, vert=False, colors=plt.colors[:len(models)])
    plt.ylabel(var_label)
    plt.title("Cross-validation: n={} k={}".format(n, k))
    if scoreindex == 0:
        plt.xlabel("End Survival Rate")
    elif scoreindex == 1:
        plt.xlabel("Median Survival Time")
    elif scoreindex == 2:
        plt.xlabel("Group size")
    elif scoreindex == 3:
        plt.xlabel("Last event time")


# In[ ]:

plot_score(repeat_results, models, 0)


# In[ ]:

plot_score(repeat_results, models, 1)


# In[ ]:

plot_score(repeat_results, models, 2)


# In[ ]:

plot_score(repeat_results, models, 3)


# In[ ]:

net = netgen(d)

