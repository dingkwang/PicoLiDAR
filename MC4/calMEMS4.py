# %%

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import glob
from sklearn.externals import joblib

dat = np.empty([1, 5]) 
for file_name in glob.glob('c*.csv'):
    d = np.empty([1, 1])
    d[0, 0] = int(file_name[1:4])
    t = np.genfromtxt(file_name,delimiter=',')
    t12 = np.mean(t[:, 0:2], axis=0, keepdims=True)
    std =  np.std(t[:, 0:2], axis=0, keepdims=True)
    tem = np.hstack((d, t12, std))
    dat = np.append(dat, tem, axis = 0)

dat = dat[1:]	
#%%

color = 'tab:red'
fig, ax1 = plt.subplots(1)
ax1.set_xlabel('Actual Distance(cm)')
ax1.set_ylabel('t(ns)', color=color)
#ax1.plot(dat[:, 0], dat[:, 1], color=color)
#ax1.plot(dat[:, 0], dat[:, 2],  color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.errorbar(dat[:, 0], dat[:, 1],  yerr=2*dat[:, 3])
#ax1.errorbar(dat[:, 0], dat[:, 2],  yerr=2*dat[:, 4], color = 'orange')

#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#color = 'tab:blue'
#ax2.set_ylabel('True Distance(cm)', color=color)  # we already handled the x-label with ax1
#ax2.plot(dat[:, 0], dat[:, 2],  color=color)
#fig.tight_layout()
#%%

#x = np.vstack((thx, thy, np.multiply(thx, thy), np.muliply(thx, thx), np.multiply(thy, thy), lpt*t, lpt*np.multiply(t, t)))
lpt = 0
#x = np.vstack((dat[:, 1], dat[:, 2]))
x = np.vstack((dat[:, 1], np.multiply(dat[:, 1], dat[:, 1])))
#x =dat[:, 1:2]
x = x.T
d = dat[:, 0:1]

#from sklearn.model_selection import train_test_split
#xtrain, xtest, dtrain, dtest = train_test_split(x, dat[:, 0], test_size=0.02, random_state=1)


#%%
poly = PolynomialFeatures(degree=1, interaction_only=True)
w = poly.fit_transform(x)
# print("w= ", w)
#%%
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(w, d)

filename = 'model2.sav'
joblib.dump(reg, filename)

#%%
WR = reg.coef_
bias = reg.intercept_
WR = np.concatenate((bias, WR), axis=None)
print("WR= ", WR)
print("bias= ", bias)

#%%

# pred = reg.predict(poly.fit_transform(x_t))
pred = reg.predict(poly.fit_transform(x))
rslt = np.hstack((pred, d))
#np.savetxt('test.csv', rslt, delimiter=',') 
print("pred vs actual ", rslt)
print("Mean squared error: %.2f"
      % mean_squared_error(d, pred))

plt.figure(3)
plt.scatter(d,pred)

