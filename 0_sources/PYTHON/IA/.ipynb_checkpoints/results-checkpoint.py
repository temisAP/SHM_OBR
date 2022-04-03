import torch
import matplotlib.pyplot as plt
import numpy as np

def results(self):

    e_T = np.zeros(len(self.X['test']))
    e_E = np.zeros(len(self.Y['test']))

    with torch.no_grad():
        self.model_T.eval()
        self.model_E.eval()
        for i in range(0,len(self.X['test'])):
            # Extract the sample
            x = torch.from_numpy( np.array([self.X['test'][i]]) ).float()
            y = self.Y['test'][i]
            # Inference
            try:
                T = self.model_T(x.to(self.device))
                E = self.model_E(x.to(self.device))
            except:
                T = self.model_T(x)
                E = self.model_E(x)
            # Error
            e_T[i] = self.scaler_T.inverse_transform(T) - self.scaler_T.inverse_transform(y[0])
            e_E[i] = self.scaler_E.inverse_transform(E) - self.scaler_T.inverse_transform(y[1])

    plt.figure()
    plt.title(f'Error absoluto medio Temperatura: {sum(abs(e_T))/len(e_T):.4f} K')
    plt.hist(e_T,bins=20)
    plt.grid()
    print(f'Error absoluto medio Temperatura: {sum(abs(e_T))/len(e_T):.4f} K ')

    plt.figure()
    plt.title(f'Error absoluto medio Deformación: {sum(abs(e_E))/len(e_E):.4f} micro')
    plt.grid()
    plt.hist(e_E,bins=20)
    print(f'Error absoluto medio Deformación: {sum(abs(e_E))/len(e_E):.4f} micro')

    plt.show()
