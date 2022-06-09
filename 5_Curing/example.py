import numpy as np
import matplotlib.pyplot as plt

line = np.linspace(0,100,100)
other_line = np.linspace(0,200,100)

data = np.random.rand(100,3)*100
other_data = np.random.rand(100,3)*100


plt.figure()
# between lines
plt.plot(line-other_line,label='between lines')
# between datas
mu = list()
for set,other_set in zip(data,other_data):
    mu.append(np.abs(np.mean(set)-np.mean(other_set)))
plt.plot(mu,label='between data')
# between line and data
mu = list()
for set,other_set in zip(data,line):
    mu.append(np.abs(np.mean(set)-np.mean(other_set)))
plt.plot(mu,label='between data and line')
plt.legend()
plt.show()
