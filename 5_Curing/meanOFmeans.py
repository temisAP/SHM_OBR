import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

file = './meanOFmeans.csv'


# Read csv with columns 'Label' and 'Value' and no index
df = pd.read_csv(file,names=['Label', 'Value'],index_col=False)

# Convert 'Value' column to list
for index, row in df.iterrows():
    row['Value'] = ast.literal_eval(row['Value'])
    n_points = len(row['Value'])

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

for index, row in df.iterrows():
    ax.plot(range(n_points),np.abs(row['Value']),'-o',label=row['Label'])

plt.xticks([y for y in range(n_points)], [y for y in range(n_points)])
ylabel = r'$-\frac{\Delta \nu}{\bar{\nu}}$'
plt.ylabel(fr'$\mu$('+ylabel+')',fontsize=10,labelpad=20).set_rotation(0)
ax.set_xlabel('Segment index')
ax.grid()
ax.legend()
locs,labels = plt.yticks()
plt.yticks(locs, map(lambda x: round(x,3), locs*1e5))
plt.text(0.0, 1.01, '1e-5', fontsize=10, transform = plt.gca().transAxes)
plt.show()
