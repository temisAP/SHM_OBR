import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

file = './meanOFmeans.csv'


# Read csv with columns 'Label' and 'Value' and no index
df = pd.read_csv(file,names=['Label', 'Value'],index_col=False)

# Convert 'Value' column to list
for index, row in df.iterrows():
    row['Value'] = ast.literal_eval(row['Value'])

plt.figure()

for index, row in df.iterrows():
    plt.plot(range(len(df)+1),row['Value'],'-o',label=row['Label'])

plt.xticks([y for y in range(len(df)+1)], [y for y in range(len(df)+1)])
plt.ylabel(r'|$\mu-\mu_0$|')
plt.xlabel('Segment index')
plt.grid()
plt.legend()
plt.show()
