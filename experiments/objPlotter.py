import matplotlib.pyplot as plt

objFile = open("objectiveStore.txt", "r")
objFile.readline()

objectiveValues = []

for line in objFile:
    objectiveValues.append([float(x) for x in line[1:-2].split(",")])


oldNumIters = 0
thresh = 0.2
for obj in objectiveValues:
    obj = [min(val, thresh) for val in obj]
    numIters = len(obj) + oldNumIters
    iterAxis = range(oldNumIters, numIters)
    oldNumIters = numIters
    plt.plot(iterAxis, obj)
    
plt.xlabel("Iterations")
plt.ylabel("Objective")
plt.title("Objective Value during Optimization")
legend = [str(2 + i) + " modes" for i in range(len(objectiveValues))]
plt.legend(legend)
plt.show()