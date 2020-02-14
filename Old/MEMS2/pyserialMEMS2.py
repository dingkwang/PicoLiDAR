import serial


filename = 'sample30_50_3.csv'
f = open(filename,'w')

with serial.Serial("COM13", 115200) as ser:
	t = 0
	for line in ser:
		if (len(line) == 14):
			a = line.decode("ascii")
			f.write(a[:-1])
			t = t+1
			if t >= 10000:
				break 
	
		