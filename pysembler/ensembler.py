#! /usr/bin/env python
import numpy as np
from sklearn import ensemble, linear_model
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import os
import sys
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)


class Ensembler(object):
    def __init__(self, model_dict, num_folds=3, task_type='classification', optimize=roc_auc_score,
                 lower_is_better=False, save_path=None):
        """
        Ensembler init function
        :param model_dict: model dictionary, see README for its format
        :param num_folds: the number of folds for ensembling
        :param task_type: classification or regression
        :param optimize: the function to optimize for, e.g. AUC, logloss, etc. Must have two arguments y_test and y_pred
        :param lower_is_better: is lower value of optimization function better or higher
        :param save_path: path to which model pickles will be dumped to along with generated predictions, or None
        """

        self.model_dict = model_dict
        self.levels = len(self.model_dict)
        self.num_folds = num_folds
        self.task_type = task_type
        self.optimize = optimize
        self.lower_is_better = lower_is_better
        self.save_path = save_path

        self.training_data = None
        self.test_data = None
        self.y = None
        self.lbl_enc = None
        self.y_enc = None

    def fit(self, training_data, y, test_data):
        """
        :param training_data: training data in tabular format
        :param y: binary, multi-class or regression
        :param test_data: test data in the same format as training data
        :return: chain of models to be used in prediction
        """

        self.training_data = training_data
        self.test_data = test_data
        self.y = y

        if self.task_type == 'classification':
            num_classes = len(np.unique(self.y))
            logger.info("Found %d classes", num_classes)
            self.lbl_enc = LabelEncoder()
            self.y_enc = self.lbl_enc.fit_transform(self.y)
            kf = StratifiedKFold(n_splits=self.num_folds)
            train_prediction_shape = (self.training_data.shape[0], num_classes)
            test_prediction_shape = (self.test_data.shape[0], num_classes)
        else:
            num_classes = -1
            self.y_enc = self.y
            kf = KFold(n_splits=self.num_folds)
            train_prediction_shape = (self.training_data.shape[0], 1)
            test_prediction_shape = (self.test_data.shape[0], 1)

        train_prediction_dict = {}
        test_prediction_dict = {}
        for level in range(self.levels):
            train_prediction_dict[level] = np.zeros((train_prediction_shape[0],
                                                     train_prediction_shape[1] * len(model_dict[level])))

            test_prediction_dict[level] = np.zeros((test_prediction_shape[0],
                                                    test_prediction_shape[1] * len(model_dict[level])))

        foldnum = 1
        for train_index, valid_index in kf.split(self.training_data, self.y_enc):
            for level in range(self.levels):
                if level == 0:
                    temp_train = self.training_data
                    temp_test = self.test_data
                else:
                    temp_train = train_prediction_dict[level - 1]
                    temp_test = test_prediction_dict[level - 1]

                for model_num, model in enumerate(self.model_dict[level]):

                    logger.info("Fold # %d. Training Level %d. Model # %d", foldnum, level, model_num)
                    model.fit(temp_train[train_index], self.y_enc[train_index])

                    logger.info("Fold # %d. Predicting Level %d. Model # %d", foldnum, level, model_num)

                    if self.task_type == 'classification':
                        temp_train_predictions = model.predict_proba(temp_train[valid_index])
                        temp_test_predictions = model.predict_proba(temp_test)
                        train_prediction_dict[level][valid_index, (model_num*num_classes):
                                                     (model_num*num_classes) + num_classes] = temp_train_predictions
                        test_prediction_dict[level][:, (model_num * num_classes):
                                                    (model_num * num_classes) + num_classes] = temp_test_predictions

                    else:
                        temp_train_predictions = model.predict(temp_train[valid_index])
                        temp_test_predictions = model.predict(temp_test)
                        train_prediction_dict[level][valid_index, model_num] = temp_train_predictions
                        test_prediction_dict[level][:, model_num] += temp_test_predictions
            foldnum += 1

        for level in range(self.levels):
            logger.info("Saving predictions for level # %d", level)
            train_predictions_df = pd.DataFrame(train_prediction_dict[level])
            test_predictions_df = pd.DataFrame(test_prediction_dict[level])
            train_predictions_df.to_csv(os.path.join(self.save_path, "train_predictions_level_" + str(level) + ".csv"),
                                        index=False, header=None)
            test_predictions_df.to_csv(os.path.join(self.save_path, "test_predictions_level_" + str(level) + ".csv"),
                                       index=False, header=None)

    def predict(self):
        pass


if __name__ == '__main__':
    model_dict = {0: [ensemble.RandomForestClassifier(),
                      ensemble.ExtraTreesClassifier(),
                      linear_model.LogisticRegression()],

                  1: [ensemble.RandomForestClassifier(),
                      ensemble.GradientBoostingClassifier()],

                  2: [linear_model.LogisticRegression()]}

    print (model_dict)
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)
    X_test = np.random.rand(500, 10)

    ens = Ensembler(model_dict=model_dict, num_folds=3, task_type='classification',
                    optimize=roc_auc_score, lower_is_better=False, save_path="../temp")
    ens.fit(X, y, X_test)
