import os
import sys
import numpy as np
from tabulate import tabulate
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.objectives import ConstrainedProblem, LeastSquaresProblem
from simsopt.solve.mpi import global_mpi_solve, least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print
from simsopt._core import Optimizable


if len(sys.argv) < 2:
    print("Error! You must specify at least 1 woutXXX.nc file.")
    exit(1)

hm = 1
hn = 0

headers = [];
vmecs = [];
quasisymmetries = ['Quasisymmetry']
aspects = ['Aspect']
iotas = ['Iota']
iters = ['Iterations']
objectives = ['Objectives']

for filename in sys.argv[1:]:
    vmecs.append(Vmec(filename, verbose=False))
    headers.append(filename[5:-14])
    iters.append(int(filename[-9:-3]))

for vmec in vmecs:
    qs = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=hm, helicity_n=hn)
    quasisymmetries.append(qs.total())
    aspects.append(vmec.aspect())
    iotas.append(vmec.mean_iota())
    prob = LeastSquaresProblem.from_tuples([(vmec.aspect, 6, 1), (vmec.mean_iota, 0.41, 1), (qs.residuals, 0, 1)])
    objectives.append(prob.objective())

print(tabulate([iters, quasisymmetries, aspects, iotas, objectives], headers=headers))





