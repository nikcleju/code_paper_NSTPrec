"""
LSIHT.py

Provides Least-Squares variant of IHT (GLSP)
"""

# Author: Nicolae Cleju
# License: BSD 3 clause

from pyCSalgos.base import SparseSolver, ERCcheckMixin

import numpy
import scipy

class NST_HT_var(ERCcheckMixin, SparseSolver):
    """
    Performs sparse coding with variants of NST_HT
    """

    # All parameters related to the algorithm itself are given here.
    # The data and dictionary are given to the run() method
    def __init__(self, variant='', eps1=1e-2, eps2=1e-2, maxiter=1000, solution_type='sparse', max_num_data=None):

        # parameter check
        if variant not in ['NST+HT', 'NST+HT+FB', 'NST+HT+subFB', 'NST+stretchedHT']:
            raise ValueError("Unknown NST_HT variant")
        self.variant = variant

        if eps1 <= 0:
            raise ValueError("eps1 is non-positive")
        self.eps1 = eps1

        if eps2 <= 0:
            raise ValueError("eps2 is non-positive")
        self.eps2 = eps2

        self.maxiter = maxiter

        if solution_type not in ['full', 'sparse', 'debias']:
            raise ValueError('unknown solution_type value')
        self.solution_type = solution_type

        self.max_num_data = max_num_data

    def __str__(self):
        return self.variant+"("+str([self.eps1, self.eps2, self.maxiter])+","+ self.solution_type +")"

    __repr__ = __str__

    def solve(self, data, dictionary, realdict=None):
        k = realdict['support'].shape[0]
        if self.max_num_data is not None:
            data = data[:, :min(self.max_num_data, data.shape[1])]
        return _NST_HT_func(data, dictionary, k, variant=self.variant, eps1=self.eps1, eps2=self.eps2, maxiter=self.maxiter, solution_type=self.solution_type)

    def checkERC(self, acqumatrix, dictoper, support):

        raise NotImplementedError('checkERC() not implemented for solver GLSP')


class StopCriterion:
    """
    Stopping criterion type:
        StopCriterion.FIXED:    fixed number of iterations
        StopCriterion.TOL:      until approximation error is below tolerance
    """
    FIXED = 1
    TOL   = 2


def _NST_HT_func(data, dictionary, k, variant, eps1=1e-2, eps2=1e-2, maxiter=1000, solution_type='sparse'):
    """
    Least Squares IHT algorihtm
    :param data: 2D array containing the data to decompose, columnwise
    :param dictionary: dictionary containing the atoms, columnwise
    :param k: sparsity level (number of non-zero coefficients)
    :param stopval: stopping criterion
    :param cond: relative condition number limit foe effective rank calculation
    :param solution_type: specifies post-processing of solution. 
        'full' means we return the full vector after the last projection step, not sparse. 
        'sparse' means we perform a final Hard Thresh of the solution. 
        'debias' means we do Hard Thdesholding and a final least-squares over the resulting support.
    :return: coefficients
    """

    # parameter check
    if variant not in ['NST+HT', 'NST+HT+FB', 'NST+HT+subFB', 'NST+stretchedHT']:
        raise ValueError("Unknown NST_HT variant")
    if eps1 < 0:
        raise ValueError("eps1 is negative")
    if eps2 < 0:
        raise ValueError("eps2 is negative")
    if maxiter < 0:
        raise ValueError("maxiter is negative")    
    if k < 0:
        raise ValueError("k is negative")
    #if k > dictionary.shape[0]:
    #    raise ValueError("k > signal size")
    if k > dictionary.shape[1]:
        raise ValueError("k > dictionary size")

    if len(data.shape) == 1:
        data = numpy.atleast_2d(data)
        if data.shape[0] < data.shape[1]:
            data = numpy.transpose(data)
    
    # Preallocate outputs
    coef = numpy.zeros((dictionary.shape[1], data.shape[1]))
    supp = []

    # Prepare the pseudoinverse of the dictionary, used for all signals
    #Dpinv_original = scipy.linalg.pinv(dictionary)    # tall matrix

    # Dimensions
    n = dictionary.shape[0]
    N = dictionary.shape[1]

    # Prepare the nullspace of the matrix
    U,S,Vt = scipy.linalg.svd(dictionary)
    eff_rank = S.size
    eff_U = U[:,:eff_rank]
    Rowsp  = Vt[:eff_rank]
    Nullsp = Vt[eff_rank:]     # last N-n rows of Vt, on rows

    for i in range(data.shape[1]):
        # Solve LS-IHT for single vector data[i]

        y = data[:,i]
        T = numpy.array(k, dtype=int)        # Set of selected atoms
        Tc = numpy.array(N-k, dtype =int)    # Set of remaining atoms

        # Starting point = Least-Squares solution
        # if cond is None:
        #     #gamma_ls = scipy.linalg.lstsq(dictionary,y)[0]  # Least-squares solution
        # else:
        #     #gamma_ls = scipy.linalg.lstsq(dictionary,y, cond=cond)[0]  # Least-squares solution
        gamma_ls =  Rowsp.T @ numpy.diag(S**(-1)) @ eff_U.T @ y  # Least-squares solution    
        gamma = gamma_ls.copy()                    # Initial gamma = least-squares solution

        # Find smallest (N-k) coefficients, in absolute values
        # - partition the indices: the indices of (N-k) smallest coefficient first
        idx = numpy.argpartition(numpy.abs(gamma), N-k)
        Tc  = idx[:(N-k)]     # holds the indices of the (N-k) smallest coefficients
        T   = idx[(N-k):]     # holds the indices of the k largest coefficients

        niter = 0;
        uk = numpy.zeros_like(gamma)
        finished = False
        while not finished:

            # 1. Sparsification
            # ==============================================================
            uk_old = uk.copy()
            uk = gamma.copy()
            uk[Tc] = 0

            # Variants:
            if variant == 'NST+HT':
                pass   # nothing else
            elif variant == 'NST+HT+FB':
                # Feedback
                uk[T] = uk[T] + scipy.linalg.lstsq(dictionary[:,T], dictionary[:,Tc] @ gamma[Tc])[0]
            elif variant == 'NST+HT+subFB':
                lmbd = 0.75 * (1/(scipy.linalg.svdvals(dictionary[:,T])[0]**2))
                uk[T] = uk[T] + lmbd*dictionary[:,T].T @ dictionary[:,Tc] @ gamma[Tc]
            elif variant ==  'NST+stretchedHT':
                theta = numpy.linalg.norm(y, 1) / numpy.linalg.norm(dictionary @ uk, 1)
                uk[T] = theta*uk[T]

            # 2. Project back on affine space of the solutions to y=D*gamma
            # ==============================================================
            gamma = gamma_ls + Nullsp.T @  Nullsp @ uk
            # Check below, might be easier
            #gamma = gamma_ls - numpy.dot(Nullsp, numpy.dot(Nullsp.T, gamma_direction)

            # 3. Update sets
            # ==================
            # Find smallest (N-k) coefficients
            # - partition the indices: the indices of (N-k) smallest coefficient first
            idx = numpy.argpartition(numpy.abs(gamma), N-k)
            Tc  = idx[:(N-k)]     # holds the indices of the (N-k) smallest coefficients
            T   = idx[(N-k):]     # holds the indices of the k largest coefficients

            niter = niter + 1

            #print(T)

            # 3. Check termination conditions
            if numpy.linalg.norm(y - dictionary[:,T] @ gamma[T],2) / numpy.linalg.norm(y) < eps1 \
              or ( numpy.linalg.norm(uk_old,2 ) > 0 and numpy.linalg.norm(uk-uk_old, 2) / numpy.linalg.norm(uk_old,2 ) < eps2) \
              or niter >= maxiter:
                finished = True

                #if variant == 'NST+HT+FB':
                #    print(niter)

        #print 'GLSP final: T = %s'%(T)

        # Final post-processing of solution:
        if solution_type == 'full':
            pass ;          # Leave gamma as is
        elif solution_type == 'sparse'    :
            gamma[Tc] = 0    # Make cosupport zero
        elif solution_type == 'debias':      
            gamma[Tc] = 0                                                              # Make cosupport zero
            gamma[T] = scipy.linalg.lstsq(dictionary[:,T], y)[0]                   # Recompute gamma on support T, since our algorithm only preserved values in Tc
        else:
            raise ValueError('Unknown value of solution_type')
        coef[:,i] = gamma
        supp.append(T)


    return coef, supp
