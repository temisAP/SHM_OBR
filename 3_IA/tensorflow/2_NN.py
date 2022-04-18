import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import pandas as pd
import random
import sys
import os

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def create_model(input_shape, num_classes):
    """ Create a 1d convolutional keras model for 2401 point signal feature extraction """

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(int(input_shape),1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    """ Train the model """
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping])
    return model

def evaluate_model(model, x_test, y_test):
    """ Evaluate the model """
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score


sys.path.append(os.path.join(os.path.dirname(__file__), '../../0_sources/PYTHON'))
from IA import IA

# %%

""" Path to dataset """

dataset = 'test_1'
path_to_dataset = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{dataset}/3_DATASET/dataset.pkl'

# %%

""" Load dataset """


IA_obj = IA('./models',name='modelos')

#IA_obj.load_datasets([path_to_dataset],val_percentage=10); IA_obj.save()
#IA_obj.load_datasets([path_to_dataset1,path_to_dataset2],plot_histogram=True,plot_preprocessing=False); IA_obj.save()

# %%

""" Create model """

input_shape = IA_obj.X['train'].shape[1]
num_classes = IA_obj.Y['train'].shape[1]

model_T = create_model(input_shape, 1)
model_E = create_model(input_shape, 1)

models = [model_T,model_E]

# %%

""" Train model """

batch_size = 32
epochs = 10

model_T = train_model(model_T,
                IA_obj.X['train'].reshape(-1,2401,1), IA_obj.Y['train'][:,0],
                IA_obj.X['val'].reshape(-1,2401,1), IA_obj.Y['val'][:,0],
                batch_size, epochs)
model_E = train_model(model_E,
                IA_obj.X['train'].reshape(-1,2401,1), IA_obj.Y['train'][:,1],
                IA_obj.X['val'].reshape(-1,2401,1), IA_obj.Y['val'][:,1],
                batch_size, epochs)

IA_obj.model_T = model_T
IA_obj.model_E = model_T

IA_obj.save_model()

# %%

""" Evaluate model """

score_T = evaluate_model(model_T, IA_obj.X['test'], IA_obj.Y['test'][:,0])
score_E = evaluate_model(model_E, IA_obj.X['test'], IA_obj.Y['test'][:,1])

# %%

""" Load model """

#IA_obj.load_model()

# %%

""" Predict """

#IA_obj.predict(model,IA_obj.X['test'],IA_obj.Y['test'])

# %%

""" Plot """

#IA_obj.plot_results()

# %%

""" Plot confusion matrix """

#IA_obj.plot_confusion_matrix()

# %%

""" Plot histogram """

#IA_obj.plot_histogram()
