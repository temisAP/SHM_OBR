import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip

def results(self,path_to = None,histograms=True,confusion=True,layers=False,representation=True,save=False):

    if path_to:
        self.X['test'] = np.loadtxt(path_to[0])
        self.Y['test'] = np.loadtxt(path_to[1])

    if self.X['test'] == 'In data loader':
        self.X['test'] = list()
        self.Y['test'] = list()
        for x,y in self.dl['test']:
            self.X['test'].append(x.cpu().detach().numpy())
            self.Y['test'].append(y.cpu().detach().numpy())


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


    """ ******** Plots ******** """

    # Ploting results keys and values (to make it easier)
    Upper_case_keys = ['Temperature', 'Deformation']
    Lower_case_keys = ['temperature', 'deformation']
    Errors          = [e_T,e_E]              if histograms else None
    Predictions     = [T_predict, E_predict] if confusion  else None
    Targets         = [T_target,  E_target]  if confusion  else None
    Unitss          = ['K','με']

    if histograms:

        from scipy.stats import norm
        import scipy.stats as st
        import statistics

        for uck, lck, data, units in zip(Upper_case_keys,Lower_case_keys,Errors,Unitss):

            print(f'*** {uck} histogram ***')

            print(f'Absolute mean error {lck}: {sum(abs(data))/len(data):.4f} {units} ')

            plt.figure()
            plt.title(f'{uck} absolute medium error: {sum(abs(data))/len(data):.4f} {units}')
            plt.hist(data,label='Data',bins=25, density=True)

            mu, std = norm.fit(data)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'tab:orange', label='Normal distribution')
            print(f'Normal distribution N(mu = {mu:.4f} {units},sigma² = {std:.4f} {units})')

            for conf_val in [0.99,0.95]:
                ci = norm.interval(conf_val, loc=mu, scale=std**0.5)
                # cnfidence interval left line
                one_x12, one_y12 = [ci[0],ci[0]], [0, np.amax(p)/2]
                # cnfidence interval right line
                two_x12, two_y12 = [ci[1],ci[1]], [0, np.amax(p)/2]
                plt.plot(one_x12, one_y12, two_x12, two_y12, marker = 'o',
                    color='tab:green' if conf_val == 0.99 else 'tab:red',
                    label=f'{int(100*conf_val)}% confident interval')
                print(f'{int(100*conf_val)}% Confident interval:[{ci[0]:.4f},{ci[1]:.4f}] {units}')

            plt.xlabel(f'{uck} error [{units}]')
            plt.ylabel('Density of results')
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.grid()
            plt.savefig(f'{save}_histogram_{lck}.png') if not save == False else False


        plt.show() if ( representation and not (confusion or layers) ) else False

    if confusion:

        for uck, lck, predict, target, units in zip(Upper_case_keys,Lower_case_keys,Predictions,Targets,Unitss):

            print(f'*** {uck} confusion ***')

            plt.figure()
            plt.title(f'Predictions vs targets: {uck}')
            plt.scatter(target,predict,label='Value')

            popt,pcov,r_squared = accuracy(target,predict)
            plt.plot(target, np.array(linear_regression(target,*popt)), color = 'tab:orange',
                     label= 'Linear regression')

            print(f'y = {popt[0]:.2f} x +{popt[1]:.2f} | r = {r_squared:.2f}')

            plt.xlabel(rf'$\Delta {uck[0]} [{units}]$ '+'(target)'     if 'K' in units else rf'$\Delta {units}$ '+'(target)')
            plt.ylabel(rf'$\Delta {uck[0]} [{units}]$ '+'(prediction)' if 'K' in units else rf'$\Delta {units}$ '+'(prediction)')
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.grid()
            plt.savefig(f'{save}_confusion_{lck}.png') if not save == False else False

        plt.show() if representation and not layers else False

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
