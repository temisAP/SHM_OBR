import torch
import matplotlib.pyplot as plt
import numpy as np

def results(self):

    e_T = np.zeros(len(self.X['test']))
    e_E = np.zeros(len(self.Y['test']))

    T_predict = list()
    T_target = list()
    E_predict = list()
    E_target = list()

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
            # Confusion
            T_predict.append(float(T.cpu().detach().numpy()))
            T_target.append(float(y[0]))
            E_predict.append(float(E.cpu().detach().numpy()))
            E_target.append(float(y[1]))

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

    plt.figure()
    plt.title('Confusion matrix: Temperatura')
    plt.scatter(T_target,T_predict)
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.grid()

    plt.figure()
    plt.title('Confusion matrix: Deformacion')
    plt.scatter(E_target,E_predict)
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.grid()

    plt.show()
