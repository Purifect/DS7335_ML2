# -*- coding: utf-8 -*-
# @Author: Puri Rudick
# @Date:   2022-01-28 18:48:16
# @Last Modified by:   Your name
# @Last Modified time: 2022-02-16 01:14:43

import numpy as np
from sklearn.metrics import accuracy_score # other metrics too pls!
from sklearn.ensemble import RandomForestClassifier # more!
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.model_selection import ParameterGrid

# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
# 2. Expand to include larger number of classifiers and hyperparameter settings
# 3. Find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function


# ************************************** #
# -----------  Example Code  ----------- #
# ************************************** #
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
      'criterion' : ['gini','entropy'],
      'min_samples_leaf': [5,10],
    }
  },
  'kNN' : {
    'model_name' : KNeighborsClassifier(),
    'param' : {
      'n_neighbors': [3,5,7,9],
      'weights': ['uniform', 'distance'],
      'algorithm': ['auto','ball_tree']
    }
  },
  'RandomForest' : {
    'model_name' : RandomForestClassifier(),
    'param' : {
      'max_depth': [5, 8, 15, 25, 30, 'None'],
      'n_estimators': [50, 150, 300, 500, 800],
      'min_samples_split': [2, 5, 10, 15, 100],
      'min_samples_leaf': [1, 2, 5, 10]
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
  ret = {} # classic explication of results

  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf
    clf.set_params(**clf_hyper) # unpack parameters into clf is they exist
    clf.fit(M[train_index], L[train_index])
    pred = clf.predict(M[test_index])
    ret[ids]= {'clf': clf,
               'parameters': clf_hyper,
               'accuracy': accuracy_score(L[test_index], pred)}
  return ret



for model_parameters in models:
  parameters = models[model_parameters]['param']
  model = models[model_parameters]['model_name']

  # build parameter grid
  sets_of_parameters = list(ParameterGrid(parameters)) # will need to be defined
  # print(sets_of_parameters)

  for parameter_set in sets_of_parameters:
    m = run(model, data, parameter_set)
    # model.set_params(**parameter_set)
    # metric = KFold(model, parameter_set, wine_df, wine_class)
    # scores.append(metric)
    print(m)

    




















# ------------------------------------------------------------ #
# 1. Write a function to take a list or dictionary of clfs and #
#    hypers(i.e. use logistic regression),                     #
#    each with 3 different sets of hyper parameters for each   #
# ------------------------------------------------------------ #

# def model_hyper():
#   models = ['DecisionTree', 'kNN', 'RandomForest']

#   classifiers = [DecisionTreeClassifier, KNeighborsClassifier, RandomForestClassifier]

#   DecisionTree_param = [{
#     'splitter': ['best', 'random'],
#     'criterion' : ['gini','entropy'],
#     'min_samples_leaf': [5,10],
#     'random_state': [0]
#   }]

#   kNN_param_ = [{ 
#     'n_neighbors': [3,5,7,9],
#     'weights': ['uniform', 'distance'],
#     'algorithm': ['auto','ball_tree', 'kd_tree', 'brute']
#   }]
  
#   RandomForest_param = [{ 
#     'max_depth': [5, 8, 15, 25, 30, 'None'],
#     'n_estimators': [50, 150, 300, 500, 800],
#     'min_samples_split': [2, 5, 10, 15, 100],
#     'min_samples_leaf': [1, 2, 5, 10]
#   }]

# # Function for clf scorer
# def scorer():
#   scorer = {
#     'accuracy': make_scorer(accuracy_score),
#     'sensitivity': make_scorer(recall_score),
#     'specificity': make_scorer(recall_score,pos_label=0)
#   }



# ----------------------------------------------------------------------------- #
# 2. Expand to include larger number of classifiers and hyperparameter settings #
# 3. Find some simple data                                                      #
# ----------------------------------------------------------------------------- #
# def model():
#   classifiers = { 'DecisionTree':[{
#                   'splitter': ['best', 'random'],
#                   'criterion' : ['gini','entropy'],
#                   'min_samples_leaf': [5,10],
#                   'random_state': [0]
#                 }]
#   }
#   return classifiers

def DecisionTree():
  param = {
    'splitter': ['best', 'random'],
    'criterion' : ['gini','entropy'],
    'min_samples_leaf': [5,10]
  }
  return param

dt = DecisionTree()
dt_keys = dt.keys()

def param_grid(model):
  param_grid = {}
  for i in range(len(model)):
    a, b = list(model.items())[i]
    param_grid = 
    
    print(b)

dt2 = param_grid(dt)


    for j in range(len(model.keys())):
      print(model.get())
      
  y=f.flatten()
  X=np.array(X)
  y=np.array(y)
  return X,y
  



def main():
  models = ['DecisionTree', 'kNN', 'RandomForest']

  classifiers = [DecisionTreeClassifier, KNeighborsClassifier, RandomForestClassifier]

  DecisionTree_param = [{
    'splitter': ['best', 'random'],
    'criterion' : ['gini','entropy'],
    'min_samples_leaf': [5,10],
    'random_state': [0]
  }]

  kNN_param = [{ 
    'n_neighbors': [3,5,7,9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto','ball_tree']
  }]
  
  RandomForest_param = [{ 
    'max_depth': [5, 10, 15, 'None'],
    'n_estimators': [50, 100, 150],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 5, 10]
  }]

# Function for clf scorer
def scorer():
  scorer = {
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0)
  }

wine_df, wine_class = datasets.load_wine(return_X_y=True)

n_folds = 5

data = (wine_df, wine_class, n_folds)

# Create function for clf models

# clf = model()

def permute_grid(grid):
  result = []
  for p in grid:
    # Always sort the keys of a dictionary, for reproducibility
    items = sorted(p.items())
    if not items:
      result = {}
    else:
      keys, values = zip(*items)
      for v in product(*values):
        params = dict(zip(keys, v))
        result.append(params)
  return result

def run(a_clf, data, clf_hyper):
    print('Classification Model: ', a_clf)
    wine_df, wine_class, n_folds = data
    kf = KFold(n_splits=n_folds)
    ret = {}

    for ids, (train_index, test_index) in enumerate(kf.split(wine_df, wine_class)):
      if a_clf == 'DecisionTree':
        clf = DecisionTreeClassifier(**param)
      # clf.fit(wine_df[train_index], wine_class[train_index])
      # pred = clf.predict(wine_df[test_index])
      # ret[ids]= {'clf': clf,
      #           'train_index': train_index,
      #           'test_index': test_index,
      #           'accuracy': accuracy_score(wine_class[test_index], pred)}
    return clf

for i in range(len(clf)):
  model, param = list(clf.items())[i]
  print(model)
  print(param)
  ret = run(model, data, param)
  print(ret)

  

# ----------------------------------------------------------------------- #
# 4. generate matplotlib plots                                            #
# that will assist in identifying the optimal clf and parampters settings #
# ----------------------------------------------------------------------- #