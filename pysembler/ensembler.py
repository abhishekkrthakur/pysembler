#! /usr/bin/env python


class Ensembler(object):

    def __init__(self, training_data, y, test_data, model_dict, num_folds, optimize, lower_is_better, save_path=None):
        """
        Ensembler init function
        :param training_data: training data in tabular format
        :param y: binary, multi-class or regression
        :param test_data: test data in the same format as training data
        :param model_dict: model dictionary, see README for its format
        :param num_folds: the number of folds for ensembling
        :param optimize: the function to optimize for, e.g. AUC, logloss, etc. Must have two arguments y_test and y_pred
        :param lower_is_better: is lower value of optimization function better or higher
        :param save_path: path to which model pickles will be dumped to along with generated predictions, or None
        """

        self.training_data = training_data
        self.y = y
        self.test_data = test_data
        self.model_dict = model_dict
        self.num_folds = num_folds
        self.optimize = optimize
        self.lower_is_better = lower_is_better
        self.save_path = save_path

