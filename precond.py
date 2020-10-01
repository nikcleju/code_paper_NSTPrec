"""
LSIHT.py

Provides Least-Squares variant of IHT (GLSP)
"""

# Author: Nicolae Cleju
# License: BSD 3 clause

from pyCSalgos.base import SparseSolver, ERCcheckMixin

import numpy
import scipy

class Preconditioner(ERCcheckMixin, SparseSolver):

    def __init__(self, solver, rcond=0, rinv=None):
        self.solver = solver

        if rcond < 0:
            raise ValueError("rcond is non-positive")
        elif rcond > 1:
            raise ValueError("rcond is larger than 1")
        self.rcond = rcond

        if rinv is not None and rinv < 0:                       # rinv is allowed to be None
            raise ValueError("rinv is non-positive")
        elif rinv is not None and rinv > 1:
            raise ValueError("rinv is larger than 1")
        elif rinv is not None and rinv < rcond:
            raise ValueError("rinv is smaller than rcond")
        self.rinv = rinv

    def solve(self, data, dictionary, realdict=None):
        # Dimensions

        # Prepare the nullspace of the matrix
        U,S,Vt = scipy.linalg.svd(dictionary)
        cond_limit = self.rcond*S[0]

        if self.rinv is None:
            # Supress all singular values smaller than limit
            eff_S = numpy.array([s for s in S if s >= cond_limit])   # singular values < cond_limit are discarded <=> set to 0
            eff_rank = eff_S.size
            eff_U = U[:,:eff_rank]
            precond_dictionary = Vt[:eff_rank,:]
            precond_data = numpy.diag(eff_S**(-1)) @ eff_U.T @ data
        else:
            # Keep all singular values, but invert only those larger than limit, and keep smaller unchanged
            inv_limit = self.rinv*S[0]
            eff_S = numpy.array([s for s in S if s >= cond_limit])   # singular values < cond_limit are discarded <=> set to 0
            inv_S = numpy.array([s**(-1) if s >= inv_limit else 1 for s in eff_S ])   # invert above inv_limit, keep until cond_limit, discard below cond_limit
            eff_rank = eff_S.size
            eff_U = U[:,:eff_rank]
            precond_dictionary  = numpy.diag(inv_S * eff_S) @ Vt[:eff_rank, :]
            precond_data = numpy.diag(inv_S) @ eff_U.T @ data
        
        return self.solver.solve(precond_data, precond_dictionary, realdict)
