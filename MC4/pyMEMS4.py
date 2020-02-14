import serial



filename = 'sm_30_40_20by20_12.5Hz_0.625Hz_Vscan2.csv'
f = open(filename,'w')

with serial.Serial("COM13", 115200) as ser:
	t = 0
	print('Serial Open')
	for line in ser:		
#		print(len(line))
		if (len(line) >= 20):
			a = line.decode("ascii")
			f.write(a[:-1])
			t = t+1
#			print(t)
			if t >= 3000:
				break 

ser.close()
f.close()
# %%

import numpy as np
import matplotlib.pyplot as plt

dat = np.empty([1, 4])
d = np.empty([1, 1])
#d = int(filename[1:3])
dat = np.genfromtxt(filename,delimiter=',')
t12 = np.mean(dat[:, 0:2], axis=0, keepdims=True)
std =  np.std(dat[:, 0:2], axis=0, keepdims=True)
time = np.max(dat[:, 2]) - np.min(dat[:, 2])
tem = np.hstack((t12, std))
#dat = np.append(dat, tem, axis = 0)

#print('n =', len(np.genfromtxt(filename,delimiter=',')))
print('ave stds:', tem )
print('time', time)	
plt.plot(dat[:, 0])
#plt.plot(dat[:, 1])