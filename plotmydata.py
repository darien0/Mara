import h5py
import matplotlib.pyplot as plt

f = h5py.File("/home/who/Documents/CALResearch/BinaryTurbulance/Mara/data/test/chkpt.0002.h5") 
rho = f['prim']['rho'][:,:,8]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(rho, interpolation='nearest')
plt.show()
f.close()
