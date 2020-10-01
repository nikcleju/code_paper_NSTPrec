"""
LSIHT.py

Provides Least-Squares variant of IHT (GLSP)
"""

# Author: Nicolae Cleju
# License: BSD 3 clause

from pyCSalgos.base import SparseSolver, ERCcheckMixin

import numpy
import scipy

class LeastSquaresIHT(ERCcheckMixin, SparseSolver):
    """
    Performs sparse coding via Least Squares IHT (LSIHT)
    """

    # All parameters related to the algorithm itself are given here.
    # The data and dictionary are given to the run() method
    def __init__(self, mu, stopval, cond=0, solution_type='sparse'):

        # parameter check
        if stopval <= 0:
            raise ValueError("stopping value is non-positive")
        if stopval < 1:
            self.stopcrit = StopCriterion.TOL
        else:
            self.stopcrit = StopCriterion.FIXED
        self.stopval = stopval

        if mu <= 0:
            raise ValueError("mu is non-positive")
        self.mu = mu

        if cond is not None and cond < 0:
            raise ValueError("cond is non-positive")
        self.cond = cond

        if solution_type not in ['full', 'sparse', 'debias']:
            raise ValueError('unknown solution_type value')
        self.solution_type = solution_type


    def __str__(self):
        return "LSIHT ("+str(self.mu)+","+str(self.stopval)+","+ str(self.cond) +")"

    __repr__ = __str__

    def solve(self, data, dictionary, realdict=None):
        k = realdict['support'].shape[0]
        return _least_squares_IHT(data, dictionary, k, self.mu, self.stopval, self.cond, self.solution_type)

    def checkERC(self, acqumatrix, dictoper, support):

        raise NotImplementedError('checkERC() not implemented for solver GLSP')

        '''
        D = numpy.dot(acqumatrix, dictoper)

        # Should normalize here the dictionary or not?
        for i in range(D.shape[1]):
            D[:,i] = D[:,i] / numpy.linalg.norm((D[:,i],2))

        dict_size = dictoper.shape[1]
        k = support.shape[0]
        num_data = support.shape[1]
        results = numpy.zeros(num_data, dtype=bool)

        for i in range(num_data):
            T = support[:,i]
            Tc = numpy.setdiff1d(range(dict_size), T)
            A = numpy.dot(D[:,Tc].T, numpy.linalg.pinv(D[:,T].T))
            assert(A.shape == (dict_size-k, k))

            linf = numpy.max(numpy.sum(numpy.abs(A),1))
            if linf < 1:
                results[i] = True
            else:
                results[i] = False
        return results
        '''

class StopCriterion:
    """
    Stopping criterion type:
        StopCriterion.FIXED:    fixed number of iterations
        StopCriterion.TOL:      until approximation error is below tolerance
    """
    FIXED = 1
    TOL   = 2


def _least_squares_IHT(data, dictionary, k, mu, stopval, cond=0, solution_type='sparse'):
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
    if stopval < 0:
        raise ValueError("stopping value is negative")
    if k < 0:
        raise ValueError("k is negative")
    if k > dictionary.shape[0]:
        raise ValueError("k > signal size")
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
    s_limit = cond*S[0]
    eff_S = numpy.array([s for s in S if s >= s_limit])
    eff_rank = eff_S.size
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
        gamma_ls =  Rowsp.T @ numpy.diag(eff_S**(-1)) @ eff_U.T @ y  # Least-squares solution    
        gamma = gamma_ls.copy()                    # Initial gamma = least-squares solution

        # Find smallest (N-k) coefficients, in absolute values
        # - partition the indices: the indices of (N-k) smallest coefficient first
        idx = numpy.argpartition(numpy.abs(gamma), N-k)
        Tc  = idx[:(N-k)]     # holds the indices of the (N-k) smallest coefficients
        T   = idx[(N-k):]     # holds the indices of the k largest coefficients

        niter = 0;
        finished = False
        while not finished:

            # 1. Take step towards minimizing the non-sparsity level
            # =======================================================

            # Zero out the largest coefficients
            gamma_direction = gamma.copy()
            gamma_direction[T] = 0

            # Take a step in new direction
            gamma_g = gamma - mu*gamma_direction

            # 2. Project back on affine space of the solutions to y=D*gamma
            # ==============================================================
            gamma = gamma_ls + numpy.dot(Nullsp.T, numpy.dot(Nullsp, gamma_g - gamma_ls))
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
            if (stopval < 1) and (numpy.linalg.norm(gamma[Tc],2) < stopval):
                finished = True
            elif (stopval > 1) and (niter == stopval):
                finished = True

        #print 'GLSP final: T = %s'%(T)

        # Final post-processing of solution:
        if solution_type == 'full':
            pass ;          # Leave gamma as is
        elif solution_type == 'sparse'    :
            gamma[Tc] = 0    # Make cosupport zero
        elif solution_type == 'debias':      
            gamma[Tc] = 0                                                              # Make cosupport zero
            if cond == None:
                gamma[T] = scipy.linalg.lstsq(dictionary[:,T], y)[0]                   # Recompute gamma on support T, since our algorithm only preserved values in Tc
            else:
                gamma[T] = scipy.linalg.lstsq(dictionary[:,T], y, cond=cond)[0]        # Also pass conditioning limit
        else:
            raise ValueError('Unknown value of solution_type')
        coef[:,i] = gamma
        supp.append(T)


    return coef, supp
