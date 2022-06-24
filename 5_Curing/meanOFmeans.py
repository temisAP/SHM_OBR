import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import matplotlib as mpl

file = './UTILS.csv'
markermap = ['-o','-v','-D','-X','-^','-s','-P']
colormap = mpl.cm.get_cmap('tab10')

# Read csv with columns 'Label' and 'Value' and no index
df = pd.read_csv(file,names=['Label', 'Value'],index_col=False)

# Convert 'Value' column to list
for index, row in df.iterrows():
    row['Value'] = ast.literal_eval(row['Value'])
    n_points = len(row['Value'])

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

i = 0
for index, row in df.iterrows():
    #ax.plot(range(n_points),row['Value'],markermap[i],label=row['Label'],color=colormap(i))

    ax.plot(range(n_points)[0:3],row['Value'][0:3],markermap[i],label=row['Label'],color=colormap(i))
    ax.plot(range(n_points)[3:],row['Value'][3:],markermap[i],label=row['Label'],color=colormap(i))
    i += 1

plt.xticks([y for y in range(n_points)], [y+1 for y in range(n_points)])
ylabel = r'$-\frac{\Delta \nu}{\bar{\nu}}$'
plt.ylabel(ylabel,fontsize=15,labelpad=20).set_rotation(0)
ax.set_xlabel('Segment index')
ax.grid()
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.tight_layout()
plt.show()
