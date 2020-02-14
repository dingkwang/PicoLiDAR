import serial


filename = 't120s.csv'
f = open(filename,'w')

with serial.Serial("COM13", 115200) as ser:
	t = 0
	print('Serial Open')
	for line in ser:		
#		print(len(line))
		if (len(line) >= 14):
			a = line.decode("ascii")
			f.write(a[:-1])
			t = t+1
#			print(t)
			if t >= 100:
				break 

ser.close()
f.close()
# %%

import numpy as np

dat = np.empty([1, 4])
d = np.empty([1, 1])
d = int(filename[1:3])
t12 = np.mean(np.genfromtxt(filename,delimiter=','), axis=0, keepdims=True)
std = np.std(np.genfromtxt(filename,delimiter=','), axis=0, keepdims=True)
tem = np.hstack((t12, std))
#dat = np.append(dat, tem, axis = 0)

print('n =', len(np.genfromtxt(filename,delimiter=',')))
print('ave stds:', tem)	
