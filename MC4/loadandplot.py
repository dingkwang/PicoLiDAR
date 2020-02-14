import numpy as np
import matplotlib.pyplot as plt

fn = "Rsm_30_40_20by20_12.5Hz_0.625Hz_Vscan2.csv"
datan = np.genfromtxt(fn ,delimiter=',')


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

s = 650
i = datan[s:, 2] -datan[s,2] 
dist = datan[s:, 3]
yyp = 1/12.5*1000
yy = (abs(((i-yyp/4) % yyp) - yyp/2)-yyp/4)/4

xxp = 1/0.625*1000
xx = (abs(((i-xxp/4) % xxp) - xxp/2)-xxp/4)/80
#fig = plt.figure(3)
#plt.plot(i, yy)

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(xx, yy, dist, c=dist, cmap='RdBu')

#for angle in range(90,180,2):
#    fig = plt.figure(4)
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter3D(xx, yy, dist, c=dist, cmap='RdBu')
#
#    ax.view_init(30,angle)
#
#    filename=str(angle)+'.png'
#    plt.savefig(filename, dpi=96)
#    plt.gca()