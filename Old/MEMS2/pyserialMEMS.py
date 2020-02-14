import serial

ser = serial.Serial('COM13', 115200, timeout = 0.1)

f = open('dataFile.txt','a')
t = 0
 

while t < 1000: 
	line = ser.readline()
	if line in ser:
		
		f.write((line.decode("ascii")).rstrip('\n'))	
		f = open('dataFile.txt','a')
		t = t+1
		print(t)

ser.close()
f.close()

#f.write(ser.readline())
#f.close()
#f = open('dataFile.txt','a')
