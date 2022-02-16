# -*- coding: utf-8 -*-
# @Author: Puri Rudick
# @Date:   2022-01-28 18:48:16
# @Last Modified by:   Your name
# @Last Modified time: 2022-02-16 01:14:43

from operator import le
import numpy as np
from sklearn.metrics import accuracy_score # other metrics too pls!
from sklearn.ensemble import RandomForestClassifier # more!
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import pandas as pd

# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
# 2. Expand to include larger number of classifiers and hyperparameter settings
# 3. Find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function


# M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
# L = np.ones(M.shape[0])
# n_folds = 5

# data = (M, L, n_folds)

# def run(a_clf, data, clf_hyper={}):
#   M, L, n_folds = data # unpack data container
#   kf = KFold(n_splits=n_folds) # Establish the cross validation
#   ret = {} # classic explication of results

#   for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
#     clf = a_clf(**clf_hyper) # unpack parameters into clf is they exist
#     clf.fit(M[train_index], L[train_index])
#     pred = clf.predict(M[test_index])
#     ret[ids]= {'clf': clf,
#                'train_index': train_index,
#                'test_index': test_index,
#                'accuracy': accuracy_score(L[test_index], pred)}
#   return ret

# results = run(RandomForestClassifier, data, clf_hyper={})
#LongLongLiveGridS#LongLon#LLongLiveGridSearch!gLiveGridSearch!


# *************************************** #
# -----------  Modified Code  ----------- #
# *************************************** #
models = {
  'DecisionTree' : {
    'model_name' : DecisionTreeClassifier(),
    'param' : {
      'splitter': ['best', 'random'],
      'criterion' : ['gini', 'entropy'],
      'min_samples_leaf': [5, 10, 20]
    }
  },
  'kNN' : {
    'model_name' : KNeighborsClassifier(),
    'param' : {
      'n_neighbors': [3, 5, 7, 9],
      'weights': ['uniform', 'distance'],
      'algorithm': ['auto','ball_tree']
    }
  },
  'RandomForest' : {
    'model_name' : RandomForestClassifier(),
    'param' : {
      'max_depth': [5, 15, 30],
      'n_estimators': [50, 150, 200],
      'min_samples_split': [2, 5, 10]
    }
  }
}

wine_df, wine_class = datasets.load_wine(return_X_y=True)
n_folds = 5
data = (wine_df, wine_class, n_folds)
scores = []


def run(a_clf, data, clf_hyper):
  M, L, n_folds = data # unpack data container
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {}
  acc = 0

  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf
    clf.set_params(**clf_hyper) # unpack parameters into clf is they exist
    clf.fit(M[train_index], L[train_index])
    pred = clf.predict(M[test_index])
    ret[ids]= {'parameters': clf_hyper}
    acc += accuracy_score(L[test_index], pred)

  acc_avg = acc / n_folds
  model_hyper_param = ret[0]
  model_hyper_param['accuracy'] = acc_avg
  return model_hyper_param

# Loop to get accuracy for each parameter of each model
for model_parameters in models:
  parameters = models[model_parameters]['param']
  model = models[model_parameters]['model_name']

  # build parameter grid
  sets_of_parameters = list(ParameterGrid(parameters))

  clf_summary = []
  for parameter_set in sets_of_parameters:
    model_ret = run(model, data, parameter_set)
    clf_summary.append(model_ret)
  
    hyper_param = []
    accuracy = []
    for i in range(len(clf_summary)):
      a, b = clf_summary[i].values()
      hyper_param.append(a)
      accuracy.append(b)

  if model_parameters in 'DecisionTree':
    DecisionTree_param = hyper_param
    DecisionTree_acc = accuracy
  elif model_parameters in 'kNN':
    kNN_param = hyper_param
    kNN_acc = accuracy
  elif model_parameters in 'RandomForest':
    RandomForest_param = hyper_param
    RandomForest_acc = accuracy

# Create histogram plot function to look for parameters that give highest accuracy
def hist(clf, param, acc):
  xs = np.arange(len(param)) 
  plt.bar(xs, acc,  align='center')
  plt.xticks(xs, param) #Replace default x-ticks with xs, then replace xs with labels
  plt.yticks(acc)
  plt.xticks(rotation='vertical')
  plt.title(clf)
  plt.show()

# Plot histogram for the 3 models
hist('DecisionTree', DecisionTree_param, DecisionTree_acc)
hist('kNN', kNN_param, kNN_acc)
hist('RandomForest', RandomForest_param, RandomForest_acc)


