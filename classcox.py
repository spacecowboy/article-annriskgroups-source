# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#from lifelines.estimation import CoxPHFitter
from pysurvival.cox import CoxModel


class CoxClasser(CoxModel):
    def __init__(self, highlim=75, lowlim=25):
        '''
        A Cox model which outputs the class of the sample:
        high, mid, or low hazard. Limits are defined as percentiles.
        '''
        self.highlim = highlim
        self.lowlim = lowlim
        super().__init__()

    def fit(self, df, duration_col, event_col=None, *args, **kwargs):
        res = super().fit(df, duration_col, event_col, *args, **kwargs)

        # Save the cut-offs for quartiles
        #preds = self.predict_partial_hazard(df).values.ravel()
        preds = self.predict(df).ravel()

        self.highcut = np.percentile(preds, self.highlim)
        self.lowcut = np.percentile(preds, self.lowlim)

        return res

    def predict_classes(self, df):
        '''
        Predict the classes of an entire DateFrame.

        Returns a DataFrame.
        '''
        #preds = self.predict_partial_hazard(df).values.ravel()
        preds = self.predict(df).ravel()
        # Remember that this is hazard and not survival
        low = (preds < self.lowcut)
        high = (preds > self.highcut)
        mid = ~high & ~low

        res = pd.DataFrame(index=df.index, columns=['group'])
        res.iloc[low, 0] = 'low'
        res.iloc[mid, 0] = 'mid'
        res.iloc[high, 0] = 'high'

        return res
