import os
import numpy as np
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.objectives import ConstrainedProblem, LeastSquaresProblem
from simsopt.solve.mpi import global_mpi_solve, least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print
from simsopt._core import Optimizable

mpi = MpiPartition()
mpi.write()

filenameLocal = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp2_QA')
filenameGlobal = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp2_QA_global')
vmecLocal = Vmec(filenameLocal, mpi=mpi, verbose=False)
vmecGlobal = Vmec(filenameGlobal, mpi=mpi, verbose=False)
#vmec.verbose = mpi.proc0_world
surfLocal = vmecLocal.boundary
surfGlobal = vmecGlobal.boundary

# Configure quasisymmetry objective:
qsLocal = QuasisymmetryRatioResidual(vmecLocal,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=0)  # (M, N) you want in |B|
qsGlobal = QuasisymmetryRatioResidual(vmecGlobal,
                                np.arange(0, 1.01, 0.1),
                                helicity_m=1, helicity_n=0)

# Define objective function


probLocal = LeastSquaresProblem.from_tuples([(vmecLocal.aspect, 6, 1), (vmecLocal.mean_iota, 0.41, 1), (qsLocal.residuals, 0, 1)])
probGlobal = LeastSquaresProblem.from_tuples([(vmecGlobal.aspect, 6, 1), (vmecGlobal.mean_iota, 0.41, 1), (qsGlobal.residuals, 0, 1)])
probGlobal = ConstrainedProblem(probGlobal.objective)


max_mode = 3

vmecLocal.indata.mpol = max_mode + 2
vmecLocal.indata.ntor = vmecLocal.indata.mpol
vmecGlobal.indata.mpol = max_mode + 2
vmecGlobal.indata.ntor = vmecGlobal.indata.mpol

# Define parameter space:
surfGlobal.fix_all()
surfGlobal.fixed_range(mmin=0, mmax=max_mode, 
                    nmin=-max_mode, nmax=max_mode, fixed=False)

surfLocal.fix_all()
surfLocal.fixed_range(mmin=0, mmax=max_mode, 
                    nmin=-max_mode, nmax=max_mode, fixed=False)

for dof in surfGlobal.local_dof_names:
    constraintSize = 0.3 if dof[5] == "1" else 0.1
    surfGlobal.set_lower_bound(key=dof, new_val=surfGlobal.get(key=dof) - constraintSize / (1.75 ** int(dof[3])))
    surfGlobal.set_upper_bound(key=dof, new_val=surfGlobal.get(key=dof) + constraintSize / (1.75 ** int(dof[3])))

    
surfLocal.fix("rc(0,0)")  # Major radius
surfGlobal.fix("rc(0,0)")


proc0_print("Beginning local optimization with max_mode =", max_mode)

#least_squares_mpi_solve(probLocal, mpi, grad=True)

proc0_print(f"Done local optimization with max_mode ={max_mode}. "
            f"Final vmec iteration = {vmecLocal.iter}")


proc0_print("Beginning global optimization with max_mode = ", max_mode)

global_mpi_solve(probGlobal, mpi, opt_method="pdfo")#, options={"ftarget" : probLocal.objective()*1.01})

proc0_print(f"Done global optimization with max_mode ={max_mode}. "
            f"Final vmec iteration = {vmecGlobal.iter}")

