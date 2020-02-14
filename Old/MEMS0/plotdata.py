import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

x = np.empty([0,1])
y = np.empty([0,1])
for file_name in glob.glob('t*.csv'):
    x = np.vstack((x, np.genfromtxt(file_name,delimiter=',')[5:50,1:2]))
    tep = np.ones([45,1])* int(file_name[1:4])
    y = np.vstack((y, tep))


dist = x*12.436-927.94
fig, ax1 = plt.subplots(1)
ax1.plot(dist)
ax1.plot(y)


print("Mean squared error: %.2f"
      % mean_squared_error(dist, y))