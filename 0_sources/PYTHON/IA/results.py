
import torch
import matplotlib.pyplot as plt
import numpy as np

def results(self):

    # Predict model
    y_pred = self.model_y.predict(self.X['test'])

    plt.figure()
    plt.title('Confusion matrix')
    plt.scatter(self.Y['test'],y_pred)
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.grid()
