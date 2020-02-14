import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

t_s = np.empty([0,1])
y = np.empty([0,1])
tm_s = np.empty([0,1])

for file_name in glob.glob('c*.csv'):
    t1t = np.genfromtxt(file_name,delimiter=',')[5:200,1:2]
    t_s = np.vstack((t_s, t1t))
    tm_s = np.vstack((tm_s, np.mean(t1t)))
    tep = np.ones([195,1])* int(file_name[1:4])
    y = np.vstack((y, tep))

print(tm_s)
#%%
dist = t_s*9.376 - 668.11
fig, ax1 = plt.subplots(1)
ax1.plot(dist)
ax1.plot(y)


print("Mean squared error: %.2f"
      % mean_squared_error(dist, y))

#%%
tt = np.genfromtxt("Scan_step1.csv",delimiter=',')[:,1:2]
dist = tt*9.376 - 667.11

fig, ax1 = plt.subplots(1)
ax1.plot(dist)