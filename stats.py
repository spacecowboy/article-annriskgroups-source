# -*- coding: utf-8 -*-
import numpy as np


def surv_area(durations, events=None, absolute=False):
    '''
    Parameters:
    durations - array of event times (must be greater than zero)
    events - array of event indicators (1/True for event, 0/False for censored)
    absolute - if True, returns the actual area. Otherwise, a relative value
               between 0 and 1

    Returns:
    area - The area under the survival curve
    '''
    if events is None:
        events = np.ones_like(durations, dtype=bool)

    events = events.astype(bool)

    # Unique event times
    TU = np.sort(np.unique(durations[events]))

    # Starting values
    S = 1.0
    A = 0.0
    p = 0
    for t in TU:
        # Add box to area
        A += S * (t - p)

        # People at risk
        R = np.sum(durations >= t)
        # Deaths between previous and now
        deaths = np.sum(durations[events] == t)

        # Update survival
        S *= (R - deaths) / R
        p = t

    # If we have censored beyond last event
    A += S * (np.max(durations) - p)

    if not absolute:
        A /= np.max(durations)

    return A


def k_fold_cross_validation(fitters, df, duration_col, event_col=None,
                            k=5, evaluation_measure=None, predictor="predict_median",
                            predictor_kwargs={}):
    """
    Perform cross validation on a dataset.
    fitter: A list of fitters. They will all train on the same subsets.
    df: a Pandas dataframe with necessary columns `duration_col` and `event_col`, plus
        other covariates. `duration_col` refers to the lifetimes of the subjects. `event_col`
        refers to whether the 'death' events was observed: 1 if observed, 0 else (censored).
    duration_col: the column in dataframe that contains the subjects lifetimes.
    event_col: the column in dataframe that contains the subject's death observation. If left
                as None, assumes all individuals are non-censored.
    k: the number of folds to perform. n/k data will be withheld for testing on.
    evaluation_measure: a function that accepts either (event_times, predicted_event_times),
                  or (event_times, predicted_event_times, event_observed) and returns a scalar value.
                  Default: statistics.concordance_index: (C-index) between two series of event times
    predictor: a string that matches a prediction method on the fitter instances. For example,
            "predict_expectation" or "predict_percentile". Default is "predict_median"
    predictor_kwargs: keyward args to pass into predictor.
    Returns:
        k-length list of scores for each fold.
    """
    n, d = df.shape
    # Each fitter has its own scores
    fitterscores = [[] for _ in fitters]

    if event_col is None:
        event_col = 'E'
        df[event_col] = 1.

    # reindex returns a copy
    df = df.reindex(np.random.permutation(df.index))
    df.sort(event_col, inplace=True)

    assignments = np.array((n // k + 1) * list(range(1, k + 1)))
    assignments = assignments[:n]

    testing_columns = df.columns - [duration_col, event_col]

    for i in range(1, k + 1):
        ix = assignments == i
        training_data = df.ix[~ix]
        testing_data = df.ix[ix]

        T_actual = testing_data[duration_col].values
        E_actual = testing_data[event_col].values
        X_testing = testing_data[testing_columns]

        for fitter, scores in zip(fitters, fitterscores):
            # fit the fitter to the training data
            fitter.fit(training_data, duration_col=duration_col, event_col=event_col)
            T_pred = getattr(fitter, predictor)(X_testing, **predictor_kwargs).values

            try:
                scores.append(evaluation_measure(T_actual, T_pred, E_actual))
            except TypeError:
                scores.append(evaluation_measure(T_actual, T_pred))

    return fitterscores
