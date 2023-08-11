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

x_test.to_csv('x_test.csv', index=False)
x_train.to_csv('x_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)