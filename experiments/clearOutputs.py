#Clears out all files from current working directory produced by VMEC, including input., wout, threed, jac_log, jxbout, mercier, and objective files.

from os import listdir, remove

filelist = listdir()
for file in filelist:
    if file[-3:] == ".nc" or file[-4:] == ".dat" or file[:6] == "input." or file[:7] == "mercier" or file[:6] == "threed":
        remove(file)