import torch
import matplotlib.pyplot as plt
import numpy as np

def results(self,histograms=True,confusion=True,layers=False,representation=True,save=False):


    print('\nTesting Neural Network')

    # Histograms
    if histograms:
        e_T = np.zeros(len(self.Y['test']))
        e_E = np.zeros(len(self.Y['test']))

    # Confusion matrix
    if confusion:
        T_predict = list()
        T_target = list()
        E_predict = list()
        E_target = list()

    if isinstance(self.model, torch.nn.Module):
        # To device
        self.model.to(self.device)

        with torch.no_grad():
            self.model.eval()
            for i in range(0,len(self.X['test'])):
                # Extract the sample
                x = torch.from_numpy( np.array([self.X['test'][i]]) ).float()
                y = self.Y['test'][i]
                # Inference
                T,E = self.model(x.to(self.device))
                # Re-scaled
                if histograms or confusion:
                    predictions = self.scalerY.inverse_transform([T,E])
                    targets     = self.scalerY.inverse_transform([y[0],y[1]])
                    try:
                        predictions = predictions.cpu().detach().numpy()
                    except:
                        pass
                    try:
                        targets = targets.cpu().detach().numpy()
                    except:
                        pass
                # Error
                if histograms:
                    e_T[i] = predictions[0][0] - targets[0][0]
                    e_E[i] = predictions[0][1] - targets[0][1]
                # Confusion
                if confusion:
                    T_predict.append(float( predictions[0][0].cpu().detach().numpy() ) )
                    T_target.append( float( targets[0][0]     ) )
                    E_predict.append(float( predictions[0][1].cpu().detach().numpy() ) )
                    E_target.append( float( targets[0][1]     ) )
    else:

        i = 0

        for x,y in zip(self.X['test'],self.Y['test']):

            # Inference
            T = self.model_T.predict(x.reshape(1, -1))
            E = self.model_E.predict(x.reshape(1, -1))
            # Error
            if histograms:
                e_T[i], e_E[i] = self.scalerY.inverse_transform([T,E]) - self.scalerY.inverse_transform([y[0],y[1]])
            # Confusion
            if confusion:
                T_predict.append(   float(self.scaler['T'].inverse_transform(    T         )))
                T_target.append(    float(self.scaler['T'].inverse_transform(    y[0]      )))
                E_predict.append(   float(self.scaler['E'].inverse_transform(    E         )))
                E_target.append(    float(self.scaler['E'].inverse_transform(    y[1]      )))

            i += 1

        T_predict = np.array(T_predict)
        T_target = np.array(T_target)
        E_predict = np.array(E_predict)
        E_target = np.array(E_target)

    if histograms:

        from scipy.stats import norm
        import statistics

        # Temperature
        plt.figure()
        plt.title(f'Temperature absolute medium error: {sum(abs(e_T))/len(e_T):.4f} K')
        plt.hist(e_T,label='Data',bins=25, density=True)

        mu, std = norm.fit(e_T)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'tab:orange', label=rf'$N(\mu = {mu:.4f},\sigma^2 = {std:.4f})$')

        plt.xlabel('Temperature error [K]')
        plt.ylabel('Density of results')
        plt.legend()
        plt.grid()

        print(f'Error absoluto medio Temperatura: {sum(abs(e_T))/len(e_T):.4f} K ')
        plt.savefig(f'{save}_histogramT.png') if not save == False else False

        # Deformation
        plt.figure()
        plt.title(rf'Deformation absolute medium error: {sum(abs(e_E))/len(e_E):.4f} $\mu \varepsilon$')
        plt.hist(e_E,label='Data',bins=25, density=True)

        mu, std = norm.fit(e_E)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'tab:orange', label=rf'$N(\mu = {mu:.4f},\sigma^2 = {std:.4f})$')

        plt.xlabel(r'Deformation error [$\mu \varepsilon$]')
        plt.ylabel('Density of results')
        plt.legend()
        plt.grid()

        print(f'Error absoluto medio Deformación: {sum(abs(e_E))/len(e_E):.4f} micro')
        plt.savefig(f'{save}_histogramE.png') if not save == False else False

        plt.show() if representation else False

    if confusion:

        plt.figure()
        plt.title('Confusion matrix: Temperature')
        plt.scatter(T_target,T_predict,label='Value')

        popt,pcov,r_squared = accuracy(T_target,T_predict)
        plt.plot(T_target, np.array(linear_regression(T_target,*popt)), color = 'tab:orange',
                 label= f'y = {popt[0]:.2f} x +{popt[1]:.2f} | r = {r_squared:.2f}')

        plt.xlabel(r'$\Delta T [K]$ '+'(target)')
        plt.ylabel(r'$\Delta T [K]$ '+'(prediction)')
        plt.legend()
        plt.grid()
        plt.savefig(f'{save}_confusionT.png') if not save == False else False

        plt.figure()
        plt.title('Confusion matrix: Deformation')
        plt.scatter(E_target,E_predict)

        popt,pcov,r_squared = accuracy(E_target,E_predict)
        plt.plot(E_target, np.array(linear_regression(E_target,*popt)), color = 'tab:orange',
                 label= f'y = {popt[0]:.2f} x +{popt[1]:.2f} | r = {r_squared:.2f}')

        plt.xlabel(r'$\Delta \: \mu \varepsilon$ '+'(target)')
        plt.ylabel(r'$\Delta \: \mu \varepsilon$ '+'(prediction)')
        plt.legend()
        plt.grid()
        plt.savefig(f'{save}_confusionE.png') if not save == False else False

        plt.show() if representation else False

    if layers:
        print('Under construction')


def linear_regression(x,a,b):
    try:
        return a*x + b
    except:
        return a*np.array(x) + b

def accuracy(xdata, ydata):
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(linear_regression, xdata, ydata)
    residuals = ydata- linear_regression(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return popt,pcov,r_squared
