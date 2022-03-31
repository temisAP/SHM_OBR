import pandas as pd
import random

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from IA.load_dataset import load_dataset

# %%

""" Load dataset """

info_file = '../1_data/dataset_information.csv'
info_df = pd.read_csv(info_file)
X,Y = load_dataset(info_df,sets = ['CM2'],leaps=[20])

# %%

""" Normalize data """

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# %%

""" Split dataset """

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_T, test_size=0.2, random_state=42)

# %%

""" Create two output model """

from keras.models import Sequential
from keras.layers import Dense, Dropout

model_T = Sequential()
model_T.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model_T.add(Dropout(0.5))
model_T.add(Dense(64, activation='relu'))
model_T.add(Dropout(0.5))
model_T.add(Dense(1, activation='sigmoid'))

model_P = Sequential()
model_P.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model_P.add(Dropout(0.5))
model_P.add(Dense(64, activation='relu'))
model_P.add(Dropout(0.5))
model_P.add(Dense(1, activation='sigmoid'))

# %%

""" Compile model """

from keras.optimizers import SGD

model_T.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])
model_P.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])

# %%

""" Fit model """

model_T.fit(X_train, Y_train[:,0], epochs=20, batch_size=32, verbose=1)
model_P.fit(X_train, Y_train[:,1], epochs=20, batch_size=32, verbose=1)

# %%

""" Evaluate model """

score_T = model_T.evaluate(X_test, Y_test[:,0], batch_size=32)
score_P = model_P.evaluate(X_test, Y_test[:,1], batch_size=32)

# %%

""" Predict model """

Y_pred_T = model_T.predict(X_test)
Y_pred_P = model_P.predict(X_test)

# %%

""" Save model """

model_T.save('./models/model_T.h5')
model_P.save('./models/model_P.h5')
