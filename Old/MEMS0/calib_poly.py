# %%

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



#dataset = np.genfromtxt('dsp.csv', delimiter=',')
dataset = np.genfromtxt('dsp.csv', delimiter=',')
# dtype = [('U8', float), ('U13', float), ('Dist', float)]
dst = np.array(dataset)
# dss = np.sort(dst, order='Dist')
# print(dss)
# np.savetxt('dss.csv', dss, delimiter=',')
#%%

#x = dataset[1:3,:]
#t = (dataset[0, :]-np.min(dataset[0, :]))/(np.max(dataset[0, :])-np.min(dataset[0, :]))
t = dataset[0, :]
thx = (dataset[1, :]-np.min(dataset[1, :]))/(np.max(dataset[1, :])-np.min(dataset[1, :]))
thy = (dataset[2, :]-np.min(dataset[2, :]))/(np.max(dataset[2, :])-np.min(dataset[2, :]))
lpt = 10
d = dataset[3, :]

color = 'tab:red'
fig, ax1 = plt.subplots(1)
ax1.set_xlabel('Samples')
ax1.set_ylabel('t(ms)', color=color)
ax1.plot(t, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('True Distance(cm)', color=color)  # we already handled the x-label with ax1
ax2.plot(d,  color=color)
fig.tight_layout()
#%%

#x = np.vstack((thx, thy, np.multiply(thx, thy), np.multiply(thx, thx), np.multiply(thy, thy), lpt*t, lpt*np.multiply(t, t)))
x = np.vstack((thx, thy, np.multiply(thx, thy), np.multiply(thx, thx), np.multiply(thy, thy), lpt*t))
x = x.T


from sklearn.model_selection import train_test_split
xtrain, xtest, dtrain, dtest = train_test_split(x, d, test_size=0.02, random_state=1)


#%%
poly = PolynomialFeatures(degree=1, interaction_only=True)
w = poly.fit_transform(xtrain)
# print("w= ", w)
#%%
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(w, dtrain)
#%%
WR = reg.coef_
bias = reg.intercept_
WR = np.concatenate((bias, WR[1:]), axis=None)
print("WR= ", WR)
print("bias= ", bias)

#%%

# pred = reg.predict(poly.fit_transform(x_t))
pred = reg.predict(poly.fit_transform(xtest))
rslt = np.vstack((pred, dtest))
np.savetxt('test.csv', rslt, delimiter=',') 
print("pred vs actual ", rslt)
print("Mean squared error: %.2f"
      % mean_squared_error(dtest, pred))

plt.figure(3)
plt.scatter(dtest,pred)
##%%
#
#plt.figure(1)
#p1 = plt.plot(t, x[:, 0], 'b', label = 't1')
#p2 = plt.plot(t, x[:, 1], 'k', label = 't2')
#plt.legend(loc='best')
#plt.xlabel('Actual Distance(cm)')
#plt.ylabel('Time (ns)')
#
#
#plt.figure(2)
#plt.scatter(t, pred)
#plt.xlabel('Actual Distance(cm)')
#plt.ylabel('Predicted Distance(cm)')
#
#
#
#valid = np.genfromtxt('m3050.csv', delimiter=',')
#predv = reg.predict(poly.fit_transform(valid))
#
#plt.figure(3)
#plt.plot(predv)
#plt.xlabel('Sequence of samples')
#plt.ylabel('Measured Distance(cm)')
#
#plt.figure(4)
#t34 = np.genfromtxt('34cm.csv', delimiter=',')
#t35 = np.genfromtxt('35cm.csv', delimiter=',')
#t36 = np.genfromtxt('36cm.csv', delimiter=',')
#p34 = reg.predict(poly.fit_transform(t34))
#p35 = reg.predict(poly.fit_transform(t35))
#p36 = reg.predict(poly.fit_transform(t36))
#plt.plot(p34, 'ko', label = '34cm')
#plt.plot(p35, 'b.', label = '35cm')
#plt.plot(p36, 'go', label = '36cm')
## print('t34.shape', t34.size)
#print('34cm std', np.std(p34))
#print('35cm std', np.std(p35))
#print('36cm std', np.std(p36))
#plt.legend(loc='best')
#plt.xlim([0, 125])
#plt.show()
#
#
## x_t = np.array([[-15.25, 23.6],
##                 [10.93, 21.25]])
## poly_x = poly.fit_transform(x)
## cal = poly_x @ WR
## print("Cal =", cal)
