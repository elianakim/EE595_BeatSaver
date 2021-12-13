import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
sys.path.append('..')


file='../20211213_f2_400hz/beat_timestamp_trajectory_dist_dynamic.csv'

BEATNUM = 3
df = pd.read_csv(file)
INDEX = [0, 397, 795, 1193,
         1592, 1989, 2388,
         2786, 3184, 3582,
         3979, 4377, 4775,
         5173, 5572, 5969] #, 6368]

if BEATNUM == 2:
    COLOR = ['r','r',
             'g','g',
             'b','b',
             'c','c',
             'y','y',
             'm','m',
             'y','y',
             'k','k',]
elif BEATNUM == 3:
    COLOR = ['r','r','r',
             'g','g','g',
             'b','b','b',
             'c','c','c',
             'y','y','y',
             'm','m','m',
             'y','y','y',
             'k','k','k',]
elif BEATNUM == 4:
    COLOR = ['r','r','r','r',
             'g','g','g','g',
             'b','b','b','b',
             'c','c','c','c',
             'y','y','y','y',
             'm','m','m','m',
             'y','y','y','y',
             'k','k','k','k']

#COLOR = np.array(list(mcolors.CSS4_COLORS.items()))[:,0]
'''
file='./20211212_f3_meta_re/beat_timestamp_trajectory_dist.csv'

INDEX = [0, 394, 692, 1090,
         1488, 1886, 2184,
         2582, 2980, 3378,
         3677, 4074, 4373]
1492    # 4
1891    # 5
2190
2589
2988
3387
3687
4085
4385
4883
5282
5582
5981
6380
6679
7178
7577
7976
8375
8674
9178
'''
'''
file = '../20211213_p3_meta/beat_timestamp_trajectory_dist.csv'
INDEX = [0, 397, 795, 1193,
         1592, 1989, 2388,
         2786, 3184, 3582,
         3979, 4377, 4775,
         5173, 5572, 5969] #, 6368]
0
398
797
1196
1596
1994
2394
2793
3192
3591
3989
4388
4787
5187
5586
5984
6383
6782
7181
7681
7980
8379
8877
9176
'''
'''
file = '../20211213_f2_400hz/beat_timestamp_trajectory_dist_dynamic.csv'
'''

fig = plt.figure()
#ax = Axes3D(fig)
ax = plt.axes(projection='3d')

#x_data = df['x'][0:10000]
#y_data = df['y'][0:10000]
#z_data = df['z'][0:10000]

x_base = 0
y_base = 0
z_base = 0

i=0
while i < (len(INDEX)-3):
    k = 0

    x_base = 0
    y_base = 0
    z_base = 0
    while k < BEATNUM:
        x_data = df['x'][INDEX[i]:INDEX[i+1]]
        y_data = df['y'][INDEX[i]:INDEX[i+1]]
        z_data = df['z'][INDEX[i]:INDEX[i+1]]

        x_data = x_data.reset_index(drop=True)
        y_data = y_data.reset_index(drop=True)
        z_data = z_data.reset_index(drop=True)

        for j in range(len(x_data)):
            x_data[j] += x_base
            y_data[j] += y_base
            z_data[j] += z_base

        x_base = x_data[len(x_data)-1].copy()
        y_base = y_data[len(x_data)-1].copy()
        z_base = z_data[len(x_data)-1].copy()

        print('from ', INDEX[i], ' to ', INDEX[i+1] ,'\n')
        print(x_data[0],', ', y_data[0], ', ', z_data[0],'\n')
        print(x_data[len(x_data)-1],', ', y_data[len(y_data)-1], ', ', z_data[len(z_data)-1],'\n')

        ax.scatter3D(x_data, y_data, z_data,  s=30, cmap=plt.cm.gray, c=COLOR[i], alpha=0.01, depthshade=False)
        print("=======================================")
        k+=1
        i+=1

#ax.view_init(60, -40)
#ax.view_init(50, -40)
#ax.view_init(40, -40)
#ax.view_init(40, -50)
#ax.view_init(10, -40)
#ax.view_init(0, -10)
#ax.view_init(30, 50)    # piano 3beats
ax.view_init(140, 90)


ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')

plt.show()
