#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pysembler
----------------------------------

Tests for `pysembler` module.
"""

from sklearn import ensemble
from sklearn.metrics import roc_auc_score
import numpy as np
import tempfile
import os

from pysembler import Ensembler

def get_num_lines_in_file(f):
    with open(f) as open_f:
        return len(open_f.readlines())

def test_the_ensembler_is_working_with_no_exceptions():
    def new_roc(y_true, y_pred):
        return roc_auc_score(y_true, y_pred[:, 1])

    model_dict = {0: [ensemble.RandomForestClassifier(n_jobs=10, n_estimators=100),
                      ensemble.ExtraTreesClassifier(n_jobs=10, n_estimators=100)],

                  1: [ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=7)]}

    X = np.random.rand(1000, 100)
    X_test = np.random.rand(100, 100)
    y = np.random.randint(0, 2, 1000)

    lentrain = X.shape[0]
    lentest = X_test.shape[0]

    train_data_dict = {0: [X, X]}
    test_data_dict = {0: [X_test, X_test]}
    with tempfile.TemporaryDirectory() as temp:
        ens = Ensembler(model_dict=model_dict, num_folds=5, task_type='classification',
                        optimize=new_roc, lower_is_better=False, save_path=temp)
        ens.fit(train_data_dict, y, lentrain)
        ens.predict(test_data_dict, lentest)
        assert get_num_lines_in_file(os.path.join(temp,'train_predictions_level_0.csv')) == lentrain
        assert get_num_lines_in_file(os.path.join(temp,'train_predictions_level_1.csv')) == lentrain
        assert get_num_lines_in_file(os.path.join(temp,'test_predictions_level_0.csv')) == lentest
        assert get_num_lines_in_file(os.path.join(temp,'test_predictions_level_1.csv')) == lentest
