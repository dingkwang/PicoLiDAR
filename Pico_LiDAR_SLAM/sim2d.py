import numpy as np
import matplotlib.pyplot as plt

'''GROUND TRUTH'''
# Target Object define by center
pcx = np.array([10])
pcy= np.array([5])
pxl = 0
pn = 10
pyl = 4

# True object position P{px, py}
px = pcx + np.linspace(-pxl/2, pxl/2, num=pn)
py = pcy + np.linspace(-pyl/2, pyl/2, num=pn)

rx = 0
ry = 0
# Robobee Position M{mx, my}
mx = 5
my = 5

plt.figure(1)
plt.plot(rx, ry, '*', color = 'k')
plt.scatter(mx, my, color = 'k')
plt.scatter(px, py, marker = '^', color = 'k')

'''Measure Data D {Theta, Distance}'''  
theta = np.arctan((py-my)/(px-mx))
d = ((px-mx)**2 + (py-my)**2)**0.5 + ((px-rx)**2 + (py-ry)**2)**0.5

c =  ((px-mx)**2 + (py-my)**2)**0.5

'''Guess Initial Position MI {mxi, myi} of RB'''
np.random.seed(1)
mxi = np.random.rand(1)*10 # Random Initial MEMS Location No1 
myi = np.random.rand(1)*10

#mxi = mx
#myi = my

'''Constructe PLD PI{pxi, pyi} based on the guess MI and measurement D{theta, d}'''

ci = (d**2-mxi**2-myi**2)/(2*mxi*np.cos(theta)+2*myi*np.sin(theta)+2*d) # Distance between MI and Object 

pxi = mxi + ci * np.cos(theta)
pyi = myi + ci * np.sin(theta)

plt.figure(1)
plt.scatter(mxi, myi, color = 'r')
plt.scatter(pxi, pyi, marker = '^', color = 'red')

#%% Initial 2
np.random.seed(2)
mxi = np.random.rand(1)*10 # Random Initial MEMS Location No1 
myi = np.random.rand(1)*10 

#mxi = mx
#myi = my

ci = (d**2-mxi**2-myi**2)/(2*mxi*np.cos(theta)+2*myi*np.sin(theta)+2*d) # Distance between MEMS and Object 

pxi = mxi + ci * np.cos(theta)
pyi = myi + ci * np.sin(theta)

plt.figure(1)
plt.scatter(mxi, myi, color = 'b')
plt.scatter(pxi, pyi, marker = '^', color = 'blue')