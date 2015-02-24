# -*- coding: utf-8 -*-
'''
Utility methods for doing survival group ensembles
'''
from ann.ensemble import Ensemble
import numpy as np
import pandas as pd


def ordered_bagging(length, count=None):
    '''Samples len elements (with replacement) from data and returns a view of
    those elements. Note that the original sorting is respected. An example
    is original list [0, 1, 2, 3, 4, 5], and result [0, 0, 1, 4, 5, 5]. Note
    the final result maintains the same sorting.
    '''
    if count is None:
        count = length
    r = np.random.randint(0, length, count)
    r.sort()  # sorts inplace
    return r


class ClassEnsemble(Ensemble):
    def __init__(self, high_nets=None, low_nets=None, netgen=None):
        '''
        Arguments, either one of these sets:
          high_nets, low_nets - lists of networks which will find high risk
                                groups and low risk groups respectively.
          netgen - A function that generates the previous two lists, with fresh networks.
        '''
        self.netgen = netgen
        if not high_nets or not low_nets:
            self.networks = []
            self.high_nets = []
            self.low_nets = []
        else:
            self.networks = high_nets + low_nets
            self.high_nets = high_nets
            self.low_nets = low_nets

            if len(high_nets) % 2 == 0 or len(low_nets) % 2 == 0:
                raise ValueError("Please supply an odd number of each network type to resolve tie issues.")
            if len(high_nets) != len(low_nets):
                raise ValueError("Please supply an equal amount of each network")

            if 2 != self.networks[0].output_count:
                raise ValueError("This class will not do what you think if networks don't have 2 output neurons.")

    def predict_class(self, indata):
        '''
        Predict the class of data point using majority voting.

        Arguments:
         indata - Data to predict

        midgroups (negative numbers) designates the (mapped) group which
        carries no real voting power. Please map them to different negative
        numbers depending on if you are selecting a low or high risk group.

        This matters is if there is a tie such as:
        5 votes 0, 5 votes 1, 0 votes 2. Having midgroup=1 here means that
        group 0 wins. Same if 1 ties with 2.

        Another example is if 0 ties with 2, as in:
        5 votes 0, 5 votes 2. In this case, the winner will be the midgroup.
        '''
        votes = {}
        hwin = None
        lwin = None

        for n in self.high_nets:
            g = 'high' if n.predict_class(indata) == 0 else 'hr'
            votes[g] = votes.get(g, 0) + 1
            if hwin is None:
                hwin = g
            elif votes[g] > votes[hwin]:
                hwin = g

        for n in self.low_nets:
            g = 'low' if n.predict_class(indata) == 0 else 'lr'
            votes[g] = votes.get(g, 0) + 1
            if lwin is None:
                lwin = g
            elif votes[g] > votes[lwin]:
                lwin = g

        if lwin == 'lr' and hwin == 'hr':
            # Answer is mid=1
            return 'mid'
        elif lwin == 'lr':
            # Answer is high risk
            return 'high'
        elif hwin == 'hr':
            # Answer is low risk
            return 'low'
        # No mid group
        elif votes[lwin] == votes[hwin]:
            # True tie, return mid
            return 'mid'
        elif votes[lwin] > votes[hwin]:
            return 'low'
        else:  # votes[hwin] > votes[lwin]
            return 'high'

    def learn(self, datain, dataout, limit=None):
        '''
        Learn using ordered bagging (maintains sort order).
        Set limit to train on smaller bagging subsets.
        Data is sorted before training.
        '''
        # First make sure data is sorted
        asc = dataout[:, 0].argsort()
        for net in self.networks:
            # Create new data using bagging. Combine the data into one array
            bag = ordered_bagging(datain.shape[0], count=limit)
            net.learn(datain[asc][bag], dataout[asc][bag])

    def fit(self, df, duration_col, event_col):
        '''
        Same as learn, but instead conforms to the interface defined by
        Lifelines and accepts a data frame as the data. Also generates
        new networks using self.netgen is it was defined.
        '''
        if self.netgen is not None:
            self.high_nets, self.low_nets = self.netgen(df)
            self.networks = self.high_nets + self.low_nets
        # Save columns for prediction later
        self.x_cols = df.columns - [duration_col, event_col]

        self.learn(df[self.x_cols].values,
                   df[[duration_col, event_col]].values)

    def predict_classes(self, df):
        '''
        Predict the classes of an entire DateFrame.

        Returns a DataFrame.
        '''
        labels, m = self.label_data(df)
        res = pd.DataFrame(index=df.index, columns=['group'])
        res.iloc[:, 0] = labels
        return res

    def label_data(self, df):
        '''
        Returns the group labels of each input pattern in the DataFrame.
        It must be a dataframe to guarantee that the same column ordering
        is used.

        Returns:
          (grouplabels, members)
        '''
        grouplabels = []
        members = {}

        for idx, tin in enumerate(df[self.x_cols].values):
            label = self.predict_class(tin)
            grouplabels.append(label)

            # Add index to member list
            if label not in members:
                members[label] = []
            members[label].append(idx)

        grouplabels = np.array(grouplabels)

        return grouplabels, members
