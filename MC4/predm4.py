# %%

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
poly = PolynomialFeatures(degree=1, interaction_only=True)
import glob
from sklearn.externals import joblib
#
t = np.empty([0,2])
y = np.empty([0,1])

#l = 1501
#for file_name in glob.glob('c*.csv'):
#    if int(file_name[1:4])>100:
#        
#        t = np.vstack((t, np.genfromtxt(file_name,delimiter=',')[1:l,0:2]))
#        print(file_name, len(t))
#        
#        tep = np.ones([l-1,1])* int(file_name[1:4])
#        y = np.vstack((y, tep))
#
#x = np.vstack((t[:, 0], np.multiply(t[:, 0], t[:, 0])))
#x = x.T
#%%
fn = "sm_30_40_20by20_12.5Hz_0.625Hz_Vscan2.csv"
data = np.genfromtxt(fn ,delimiter=',')[1:,:]
t = np.vstack((t, np.genfromtxt(fn ,delimiter=',')[1:,0:2]))
xt = np.vstack((t[:, 0], np.multiply(t[:, 0], t[:, 0])))
xt = xt.T

#%%
filename = 'model2.sav'
reg = joblib.load(filename)

poly = PolynomialFeatures(degree=1, interaction_only=True)
pred = reg.predict(poly.fit_transform(xt))

#fig, ax1 = plt.subplots(1)
#ax1.plot(pred)
#ax1.plot(y)
#print("Mean squared error: %.2f"
#      % mean_squared_error(y, pred))
#%%



datan = np.hstack((data, pred))
np.savetxt("Rsm_30_40_20by20_12.5Hz_0.625Hz_Vscan2.csv", datan, delimiter=',') 

#%%
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

s = 650
i = datan[s:, 2] -datan[s,2] 
dist = datan[s:, 3]
yyp = 1/12.5*1000
yy = (abs(((i-yyp/4) % yyp) - yyp/2)-yyp/4)/4

xxp = 1/0.625*1000
xx = (abs(((i-xxp/4) % xxp) - xxp/2)-xxp/4)/80
fig = plt.figure(3)
plt.plot(i, yy)

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(xx, yy, dist, c=dist, cmap='RdBu')