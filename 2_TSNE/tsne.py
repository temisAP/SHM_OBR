import numpy as np
from scipy import signal
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from IA import IA


dataset = 'test_1'
path_to_dataset = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{dataset}/3_DATASET/dataset.pkl'

# %%

""" Load IA """

IA_obj = IA('./models',name=dataset)
IA_obj.save()

#IA_obj.load_datasets(path_to_dataset,plot_histogram=True,plot_preprocessing=True)
IA_obj.load_datasets(path_to_dataset,split=False, preprocessing=False)

# %%

IA_obj.save()

# %%

""" Get SS """

ss    = np.array([lst[15] for lst in IA_obj.X]) # 0-15
temps = np.array([lst[0] for lst in IA_obj.Y])
defs  = np.array([lst[1] for lst in IA_obj.Y])

# %%

# Plot spectralshift
fig, ax = plt.subplots(1,2)

ax[0].scatter(ss,defs, c=defs, cmap='jet')
sm0 = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=min(defs), vmax=max(defs)))
cbar0 = plt.colorbar(sm0,ax=ax[0],spacing='proportional')
cbar0.set_label(r'$\mu \varepsilon$',rotation=0,labelpad=10)
ax[0].set_xlabel('Spectralshift')
ax[0].set_ylabel(r'$\mu \varepsilon$')


ax[1].scatter(ss, temps, c=temps, cmap='jet')
sm1 = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=min(temps), vmax=max(temps)))
cbar1 = plt.colorbar(sm1,ax=ax[1], spacing='proportional')
cbar1.set_label(r'$\Delta$ T [K]',rotation=0,labelpad=10)
ax[1].set_xlabel('Spectralshift')
ax[1].set_ylabel(r'$\Delta$ T [K]')

ax[0].grid()
ax[1].grid()
plt.show()

# %%

""" TSNE """

# Fit the model
#model = TSNE(n_components=2, perplexity=50.0, early_exaggeration=15.0,learning_rate=80)
model = PCA(n_components=2)
np.set_printoptions(suppress=True)
Y =  model.fit_transform(IA_obj.X)


# Plot the data
fig, ax = plt.subplots(1,2)

ax[0].scatter(Y[:,0],Y[:,1], c=defs, cmap='jet')
sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=min(defs), vmax=max(defs)))
cbar = plt.colorbar(sm,ax=ax[0],spacing='proportional')
cbar.set_label(r'$\mu \varepsilon$',rotation=0,labelpad=10)

ax[1].scatter(Y[:,0],Y[:,1], c=temps, cmap='jet')
sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=min(temps), vmax=max(temps)))
cbar = plt.colorbar(sm,ax=ax[1], spacing='proportional')
cbar.set_label(r'$\Delta$ T [K]',rotation=0,labelpad=10)

ax[0].grid()
ax[1].grid()
plt.show()
