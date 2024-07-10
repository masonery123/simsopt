import os
import numpy as np
import matplotlib.pyplot as plt
from simsopt.util import MpiPartition
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve

mpi = MpiPartition(ngroups=4)

filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp2_QA_000_000394')
equil = Vmec(filename, mpi)
surf = equil.boundary

surf.fix_all()
surf.unfix('rc(1,1)')
surf.unfix('zs(1,1)')

qs = QuasisymmetryRatioResidual(equil, np.arange(0, 1.01, 0.2), helicity_m = 1, helicity_n = 0)

#(equil.iota_axis, 0.41, 1), (equil.volume, 0.15, 1)
#,(qs.residuals, 0, 0.0001)
prob = LeastSquaresProblem.from_tuples([(qs.residuals, 0, 1)])

shiftr = 0
shiftz = 0
lim = 0.2
res = 16
cutoff = 0.5

Rs = np.arange(shiftr-lim, shiftr+lim, lim/res)
Zs = np.arange(shiftz-lim, shiftz+lim, lim/res)
out = np.zeros((Rs.size, Zs.size))
for rInd in range(Rs.size):
    for zInd in range(Zs.size):
        #out[rInd, zInd] = prob.objective(x=(Rs[rInd], Zs[zInd]))
        if Zs[zInd] + Rs[rInd] >= -0.02 and Rs[rInd] >= -0.02:
            try: 
                out[rInd, zInd] = prob.objective(x=(Rs[rInd], Zs[zInd]))
                if out[rInd, zInd] > cutoff:
                    out[rInd, zInd] = 0
                else:
                    out[rInd, zInd] = cutoff - out[rInd, zInd]
            except:
                out[rInd, zInd] = 0
        else:
            out[rInd, zInd] = 0

out = np.flipud(out)

plt.imshow(out, extent=[shiftr-lim, shiftr+lim, shiftz-lim, shiftz+lim])
plt.xlabel("RC(1,1)")
plt.ylabel("ZS(1,1)")
plt.title("Objective Landscape near QS")
plt.colorbar()
plt.show()