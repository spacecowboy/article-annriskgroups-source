# -*- coding: utf-8 -*-
import numpy as np


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
    df = df.copy()

    if event_col is None:
        event_col = 'E'
        df[event_col] = 1.

    df = df.reindex(np.random.permutation(df.index)).sort(event_col)

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
