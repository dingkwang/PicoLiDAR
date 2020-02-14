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
file_name = 


x = np.vstack((x, np.genfromtxt(file_name,delimiter=',')[1:l,:]))



#%%
from sklearn.linear_model import LinearRegression
filename = 'model1.sav'
reg = joblib.load(filename)

poly = PolynomialFeatures(degree=1, interaction_only=True)
pred = reg.predict(poly.fit_transform(x))

fig, ax1 = plt.subplots(1)
ax1.plot(pred)

