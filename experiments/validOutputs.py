#Prints a list of VMEC outputs that converged and clears out input. and wout files from failed VMEC runs in the current directory.

from os import listdir, remove
from scipy.io import netcdf_file

goodInputs = []
filelist = listdir()
for file in filelist:
    if file[:4] == "wout":
        try:
            f = netcdf_file(file,'r',mmap=False)
            phi = f.variables['phi'][()]
            print(file)
            goodInputs.append("input." + file[5:-3])
        except:
            remove(file)
for file in filelist:
    if file[:6] == "input." and not file in goodInputs:
        remove(file)