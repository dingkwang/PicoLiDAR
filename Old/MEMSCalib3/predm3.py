# %%

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
poly = PolynomialFeatures(degree=1, interaction_only=True)
import glob
from sklearn.externals import joblib

x = np.empty([0,2])
y = np.empty([0,1])
l = 2001 
for file_name in glob.glob('c*.csv'):
    x = np.vstack((x, np.genfromtxt(file_name,delimiter=',')[1:l,:]))
    tep = np.ones([l-1,1])* int(file_name[1:4])
    y = np.vstack((y, tep))

x = x

#%%
from sklearn.linear_model import LinearRegression
filename = 'model1.sav'
reg = joblib.load(filename)

poly = PolynomialFeatures(degree=1, interaction_only=True)
pred = reg.predict(poly.fit_transform(x))

fig, ax1 = plt.subplots(1)
ax1.plot(pred)
ax1.plot(y)

print("Mean squared error: %.2f"
      % mean_squared_error(y, pred))