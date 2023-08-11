import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("douanesDataset.csv")

df = data.drop_duplicates()
min_count=df['ILLICIT'].value_counts().min()

balanced_df = df.groupby('ILLICIT').apply(lambda x: x.sample(min_count)).reset_index(drop=True)

df.dropna(inplace=True)

df.drop(['YEAR','HS4', 'HS2', 'HS6.Origin', 'TARIFF.CODE', 'DECLARANT.CODE','OFFICEIMPORTER.TIN', 'SGD.MonthofYear'], axis=1, inplace=True)

min_count=df['ILLICIT'].value_counts().min()

balanced_df = df.groupby('ILLICIT').apply(lambda x: x.sample(min_count)).reset_index(drop=True)

labels=balanced_df["ILLICIT"]

features=balanced_df.drop(['ILLICIT'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

scaler = StandardScaler()

scaler.fit(x_train)

x_train_norm = scaler.transform(x_train)
x_test_norm = scaler.transform(x_test)

def gridsearch_dit(models, params,train_x, train_y):
  model_best=[]
  accuracy_best=[]
  for i in range(len(models)):
    print(f'tour numero: {i}')
    model=GridSearchCV(models[i], params[i], cv=5)
    model.fit(train_x, train_y)
    print(model.best_estimator_)
    print(model.best_score_)
    model_best.append(model.best_estimator_)
    accuracy_best.append(model.best_score_)
  return model_best, accuracy_best

param_grid_rf = [
    {'n_estimators':[10,30]},
    {'max_depth':[2,5,7]}, 
    {'max_features':[2,4,6]}, 
    {'min_samples_leaf':[2,5,9,3]}
]
param_grid_knn= [
    {'n_neighbors':[3,5,7,10,15,12,13,16,19,17,11]}
]
param_grid_log=[
    {'max_iter':[10,100,120,5,90,80,85]}
]

classifiers =[]
classifiers.append(RandomForestClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())

parametres=[]
parametres.append(param_grid_rf)
parametres.append(param_grid_knn)
parametres.append(param_grid_log)

best_model= gridsearch_dit(classifiers, parametres, X_train, Y_train)
mod, accu = best_model
for i in range(len(mod)):
  if accu[i]== max(accu):
    mod[i].save('model.h5')