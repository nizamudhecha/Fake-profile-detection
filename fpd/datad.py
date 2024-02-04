import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_curve, confusion_matrix

import os

# List files in the 'fpd/input' directory
for dirname, _, filenames in os.walk('/fpd/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings("ignore")

# Define a function to filter deprecation warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

# Use simplefilter to ignore deprecation warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    fxn()

# Load the training dataset
instagram_df_train=pd.read_csv('fpd/train.csv')
#print(instagram_df_train)
instagram_df_test=pd.read_csv('fpd/test.csv')
"""print(instagram_df_test)
instagram_df_train.head()
instagram_df_train.tail()
instagram_df_train.info()
print(instagram_df_train.describe())
print(instagram_df_train.isnull().sum())
print(instagram_df_train['profile pic'].value_counts())
print(instagram_df_train['fake'].value_counts())
sns.countplot(instagram_df_train['fake'])
plt.show()
sns.countplot(instagram_df_train['private'])
plt.show()
sns.countplot(instagram_df_train['profile pic'])
plt.show()
plt.figure(figsize = (20, 10))
sns.distplot(instagram_df_train['nums/length username'])
plt.show()
plt.figure(figsize=(20, 20))
cm = instagram_df_train.corr()
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)
plt.show()"""""

X_train = instagram_df_train.drop(columns = ['fake'])
X_test = instagram_df_test.drop(columns = ['fake'])
print(X_train)


y_train = instagram_df_train['fake']
y_test = instagram_df_test['fake']
print(y_train)

# Scale the data before training the model
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

y_train = tf.keras.utils.to_categorical(y_train, num_classes = 2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes = 2)

print(y_train)

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(50, input_dim=11, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation='softmax'))

model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
epochs_hist = model.fit(X_train, y_train, epochs = 50,  verbose = 1, validation_split = 0.1)