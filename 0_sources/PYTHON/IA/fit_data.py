from statistics import mean
import torch
import torch.nn as nn
import time
from .model import splitter
import numpy as np


def fit_data(self,num_epochs = 25):

    print('\nFitting data')

    # Compile model
    model = splitter()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Fit model
    model.fit(self.X['train'], self.Y['train'],
              batch_size=32, epochs=num_epochs, verbose=1)

    # Evaluate model
    score = model.evaluate(self.X['test'], self.Y['test'], verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    self.model_y = model
    return model
