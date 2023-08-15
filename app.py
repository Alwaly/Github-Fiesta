import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import zipfile
import os

os.makedirs('datasets')
data = pd.read_csv("douanesDataset.csv")

df = data.drop_duplicates()
df.dropna(inplace=True)

df.drop(['YEAR'], axis=1, inplace=True)

min_count=df['ILLICIT'].value_counts().min()

df = df.groupby('ILLICIT').apply(lambda x: x.sample(min_count)).reset_index(drop=True)

Y = df['ILLICIT']
X = df.drop('ILLICIT', axis=1)

continuous_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

df_continuous = X[continuous_features]

df_categorical = X[categorical_features]

X_ = X.drop(categorical_features, axis=1)
X = X.drop(continuous_features, axis=1)

label_encoder = LabelEncoder()

for column in categorical_features:
    X[column] = label_encoder.fit_transform(X[column])
  
X_ = StandardScaler().fit_transform(X_)

X_ = pd.DataFrame(X_, columns=continuous_features)
X = pd.concat([X_,X], axis=1, ignore_index=True)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

x_train.to_csv('datasets/x_train.csv', index=False)
y_train.to_csv('datasets/y_train.csv', index=False)
x_test.to_csv('datasets/x_test.csv', index=False)
y_test.to_csv('datasets/y_test.csv', index=False)



folder = 'datasets'
output = f'{folder}.zip'
with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(folder):
      for file in files:
          file_path = os.path.join(root, file)
          arcname = os.path.relpath(file_path, folder)
          zipf.write(file_path, arcname)


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

best_model= gridsearch_dit(classifiers, parametres, x_train, y_train)

mod, accu = best_model
for i in range(len(mod)):
  if accu[i]== max(accu):
    with open("model.pkl", 'wb') as model_file:
      pkl.dump(mod[i], model_file)