import h5py
import math
import numpy as np
import os
import scipy.io as sio
import spams

from collections import namedtuple

import sys, socket
hostname = socket.gethostname()
if hostname == 'caraiman':
    pyCSalgos_path = '/home/nic/code/pyCSalgos'
elif hostname == 'nclejupc':
    pyCSalgos_path = '/home/ncleju/code/pyCSalgos'
elif hostname == 'nclejupchp':
    pyCSalgos_path = '/home/ncleju/Work/code/pyCSalgos'
#sys.path.insert(0,'/home/ncleju/code/pyCSalgos')
#sys.path.append('D:\\Facultate\\Code\\pyCSalgos')
sys.path.append(pyCSalgos_path)

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

#from pyCSalgos import AnalysisPhaseTransition
#from pyCSalgos import UnconstrainedAnalysisPursuit
#from pyCSalgos import AnalysisBySynthesis
#from pyCSalgos import OrthogonalMatchingPursuit
from LSIHT import LeastSquaresIHT
from NST import NST_HT_var
from pyCSalgos import SynthesisPhaseTransition
from pyCSalgos import IterativeHardThresholding
from pyCSalgos import OrthogonalMatchingPursuit
from pyCSalgos import ApproximateMessagePassing
from precond import Preconditioner


plot_options = {                 # For all algorithms, in any order. Only the used ones are taken.
        'IHTa':             {'color': 'tab:blue',   'marker': 'x', 'linestyle':'-'}, 
        'IHTmu1':           {'color': 'tab:orange', 'marker': '+', 'linestyle':'-'}, 
        'OMP':              {'color': 'tab:green',  'marker': '1',   'linestyle':'-'},
        'AMP':              {'color': 'tab:red',    'marker': '2',   'linestyle':'-'},
        'NST+HT+FB':        {'color': 'tab:purple', 'marker': '^', 'linestyle':'-'},
        'NST+HT+subFB':     {'color': 'tab:brown',  'marker': 'v', 'linestyle':'-'},
        'NST+stretchedHT':  {'color': 'tab:pink',   'marker': '<', 'linestyle':'-'},

        'P-IHTa':           {'color': 'tab:blue',   'marker': '+', 'linestyle':'--'},
        'P-IHTmu1':         {'color': 'tab:orange', 'marker': '1', 'linestyle':'--'}, 
        'P-OMP':            {'color': 'tab:green',  'marker': '2', 'linestyle':'--'}, 
        'P-AMP':            {'color': 'tab:red',    'marker': '^', 'linestyle':'--'}, 
        'P-NST+HT+FB':        {'color': 'tab:purple', 'marker': 'v', 'linestyle':'--'},
        'P-NST+HT+subFB':     {'color': 'tab:brown',  'marker': '<', 'linestyle':'--'},
        'P-NST+stretchedHT':  {'color': 'tab:pink',   'marker': 'x', 'linestyle':'--'},       
        }
        
#==============================================================================
# Named tuple to define parameters
class ParamsSC(object):
    def __init__(self, **kwargs):

        # Default parameters first
        self.snr_db_sparse = np.Inf
        self.snr_db_signal = np.Inf
        self.snr_db_signal = 1e-6
        self.save_folder = 'save/SparseCoding'
        self.plot_options = plot_options
        self.ylim = (None, None)

        # Add / overwrite with the specified parameters
        for (k,v) in kwargs.items():
            setattr(self, k, v)

#==============================================================================


def plotdecays(length, As, RCconsts, filename):

    plt.figure()
    plt.gca().set_prop_cycle(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
                            linestyle=['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--'])
    
    legend = []
    for (A, RCconst) in zip(As, RCconsts):
        S = A * np.exp(-RCconst / length * np.arange(length))
        S = S / np.linalg.norm(S) * math.sqrt(length)
        plt.plot(S)
        #plt.show()
        legend.append('Constant' if RCconst == 0 else r'Exponential decay $\tau={:.1f}$'.format(RCconst))
    #plt.legend(['Constant', 'Exponential decay $\tau=0.007$', 'Exponential decay $\tau=0.01$'])
    plt.legend(legend)
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.savefig(filename + '.png', bbox_inches='tight')
    plt.close()


def dictdecay(dictionary, A, RCconst, normed=True, plotS=None):
    # Apply exponential decay to singular values
    [U,S,Vt] = np.linalg.svd(dictionary, full_matrices=False)
    n = dictionary.shape[0]
    S = A * np.exp(-RCconst / n * np.arange(S.size))
    S = S / np.linalg.norm(S) * math.sqrt(dictionary.shape[0])
    if plotS:
        plt.plot(S)
        #plt.show()
        plt.savefig(plotS + '_spectrum' + '.' + 'pdf', bbox_inches='tight')
        plt.savefig(plotS + '_spectrum' + '.' + 'png', bbox_inches='tight')
        plt.close()
    dictionary = U @ np.diag(S) @ Vt
    cond_beforenorm = S[0] / S[-1]              # condition number before normalization

    for i in range(dictionary.shape[1]):
        dictionary[:,i] = dictionary[:,i] / np.sqrt(np.sum( dictionary[:,i]**2 ))  # normalize columns

    S = np.linalg.svd(dictionary, compute_uv=False)
    cond_afternorm = S[0] / S[-1]              # condition number before normalization
    print('Condition number: {:2f}, before normalization was {:2f}'.format(cond_afternorm, cond_beforenorm))
    return dictionary

def make_params_randn(signal_size, dict_size, RCconst, snr_db_signal, snr_db_sparse, num_data=100, seed=None, rhomax=1, specialname=None, rcond=0, rinv=None, ylim=(None, None)):

    #=============================================================
    # Use a random dictionary with decaying spectrum, normalized
    
    p = ParamsSC()
    p.name         = 'SC{}_randn_{}_{}x{}_{:.3f}'.format('_'+specialname if specialname is not None else '', snr_db_signal, signal_size, dict_size, RCconst).replace('.','p')
    p.rhos         = np.arange(0.05, rhomax, 0.05)     # Phase transition grid, y axis
    #snr_db_sparse  = np.Inf                      # SNR ratio
    #solvers        = [LeastSquaresIHT(1, 1000, solution_type='sparse', cond=1e-3), IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), OrthogonalMatchingPursuit(1e-7), ApproximateMessagePassing(1e-7, 1000)]
    p.solvers      = [  #LeastSquaresIHT(1, 1000, solution_type='sparse', cond=0), 
                        IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), 
                        IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), 
                        OrthogonalMatchingPursuit(1e-7, algorithm='sparsify_QR'),
                        ApproximateMessagePassing(1e-7, 1000), 
                        Preconditioner(IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), rcond=rcond, rinv=rinv),
                        Preconditioner(IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), rcond=rcond, rinv=rinv),
                        Preconditioner(OrthogonalMatchingPursuit(1e-7, algorithm='sparsify_QR'), rcond=rcond, rinv=rinv),
                        Preconditioner(ApproximateMessagePassing(1e-7, 1000), rcond=rcond, rinv=rinv),
                        #NST_HT_var('NST+HT+FB', 1e-2, 1e-2, 1000),
                        NST_HT_var('NST+HT+subFB', 1e-2, 1e-2, 1000),
                        NST_HT_var('NST+stretchedHT', 1e-2, 1e-2, 1000),                        
                     ]
    # solver_names   = [  'IHTa',             # Names used for figure files
    #                     'IHTmu1', 
    #                     'OMP', 
    #                     'AMP', 
    #                     'IHTa-P', 
    #                     'IHTmu1-P', 
    #                     'OMP-P', 
    #                     'AMP-P',
    #                      #'NST+HT+FB',
    #                     'NST+HT+subFB',
    #                     'NST+stretchedHT',                        
    #                     ]    
    p.solver_names = [get_solver_shortname(solver) for solver in p.solvers]
    p.success_thresh = 1e-6 if snr_db_signal == np.Inf else None      # Threshold for considering successful recovery
    if seed != None:
        np.random.seed(seed)
    p.dictionary     = dictdecay(dictionary=np.random.randn(signal_size, dict_size), A=1, RCconst=RCconst, normed=True, plotS=p.save_folder+'/'+p.name)
    p.snr_db_signal = snr_db_signal
    p.snr_db_sparse = snr_db_sparse
    p.num_data = num_data
    p.ylim = ylim
    #=============================================================

    return p

def make_params_ECG(signal_size, dict_size, snr_db_signal, snr_db_sparse, num_data=100, seed=None, rhomax=1, specialname=None, rcond=0, rinv=None, ylim=(None, None)):

    #=============================================================
    # Use a random dictionary with decaying spectrum, normalized
    
    p = ParamsSC()
    p.name         = 'SC{}_ECG_{}_{}x{}'.format('_'+specialname if specialname is not None else '', snr_db_signal, signal_size, dict_size).replace('.','p')
    p.rhos         = np.arange(0.05, rhomax, 0.05)     # Phase transition grid, y axis
    #snr_db_sparse  = np.Inf                      # SNR ratio
    # solvers        = [LeastSquaresIHT(1, 1000, solution_type='sparse'), IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), OrthogonalMatchingPursuit(1e-7), ApproximateMessagePassing(1e-7, 1000)]
    # solver_names   = ['IAP','IHTa', 'IHTmu1', 'OMP', 'AMP']    # Names used for figure files
    p.solvers      = [  #LeastSquaresIHT(1, 1000, solution_type='sparse', cond=0), 
                    IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), 
                    IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), 
                    OrthogonalMatchingPursuit(1e-7, algorithm='sparsify_QR'),
                    ApproximateMessagePassing(1e-7, 1000), 
                    Preconditioner(IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), rcond=rcond, rinv=rinv),
                    Preconditioner(IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), rcond=rcond, rinv=rinv),
                    Preconditioner(OrthogonalMatchingPursuit(1e-7, algorithm='sparsify_QR'), rcond=rcond, rinv=rinv),
                    Preconditioner(ApproximateMessagePassing(1e-7, 1000), rcond=rcond, rinv=rinv),
                    #NST_HT_var('NST+HT+FB', 1e-2, 1e-2, 1000),
                    NST_HT_var('NST+HT+subFB', 1e-2, 1e-2, 1000),
                    NST_HT_var('NST+stretchedHT', 1e-2, 1e-2, 1000),                        
                    ]
    p.solver_names = [get_solver_shortname(solver) for solver in p.solvers]
    p.success_thresh = 1e-6 if snr_db_signal == np.Inf else None      # Threshold for considering successful recovery
    if seed != None:
        np.random.seed(seed)
    # Load dictionary
    dictfile = 'dicts/ECGdict{}x{}.npz'.format(signal_size, dict_size)    
    print('Loading dictionary from {}'.format(dictfile))
    npzfile = np.load(dictfile)
    p.dictionary = npzfile['arr_0']
    print("Loaded dictionary with size = {}x{}".format(p.dictionary.shape[0], p.dictionary.shape[1]))
    p.snr_db_signal = snr_db_signal
    p.snr_db_sparse = snr_db_sparse
    p.num_data = num_data   
    p.ylim = ylim 
    #=============================================================

    return p

def make_params_ECG_onlyAMPP_partcond(*args, **kwargs):
    p = make_params_ECG(*args, **kwargs)
    # Overwrite solvers
    p.solvers = [   Preconditioner(ApproximateMessagePassing(1e-7, 1000), rcond=kwargs['rinv'], rinv=kwargs['rinv']),
                    Preconditioner(ApproximateMessagePassing(1e-7, 1000), rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(ApproximateMessagePassing(1e-7, 1000), rcond=kwargs['rcond'], rinv=kwargs['rcond'])
                    ]
    p.solver_names = [  'Partial precond (rcond = {}, rinv = {})'.format(kwargs['rinv'], kwargs['rinv']),
                        'Partial precond (rcond = {}, rinv = {})'.format(kwargs['rcond'], kwargs['rinv']),
                        'Partial precond (rcond = {}, rinv = {})'.format(kwargs['rcond'], kwargs['rcond'])
                        ]
    p.plot_options = None
    return p

def make_params_ECG_onlyNSTHTFB(*args, **kwargs):
    p = make_params_ECG(*args, **kwargs)
    # Overwrite solvers
    p.solvers = [   NST_HT_var('NST+HT+FB', 1e-2, 1e-2, 1000),
                    ]
    p.solver_names = [get_solver_shortname(solver) for solver in p.solvers]
    return p

def make_params_ECG_allsolvers(*args, **kwargs):
    p = make_params_ECG(*args, **kwargs)
    # Overwrite solvers
    p.solvers      = [
                    IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), 
                    IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), 
                    OrthogonalMatchingPursuit(1e-7, algorithm='sparsify_QR'),
                    ApproximateMessagePassing(1e-7, 1000), 
                    NST_HT_var('NST+HT+FB', 1e-2, 1e-2, 1000),
                    NST_HT_var('NST+HT+subFB', 1e-2, 1e-2, 1000),
                    NST_HT_var('NST+stretchedHT', 1e-2, 1e-2, 1000),                        
                    Preconditioner(IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(OrthogonalMatchingPursuit(1e-7, algorithm='sparsify_QR'),                                rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(ApproximateMessagePassing(1e-7, 1000),                                                   rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(NST_HT_var('NST+HT+FB', 1e-2, 1e-2, 1000),                                               rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(NST_HT_var('NST+HT+subFB', 1e-2, 1e-2, 1000),                                            rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(NST_HT_var('NST+stretchedHT', 1e-2, 1e-2, 1000),                                         rcond=kwargs['rcond'], rinv=kwargs['rinv'])
                    ]
    p.solver_names = [get_solver_shortname(solver) for solver in p.solvers]
    return p


def make_params_randn_allsolvers(*args, **kwargs):
    p = make_params_randn(*args, **kwargs)
    # Overwrite solvers
    p.solvers      = [
                    IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), 
                    IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), 
                    OrthogonalMatchingPursuit(1e-7, algorithm='sparsify_QR'),
                    ApproximateMessagePassing(1e-7, 1000), 
                    NST_HT_var('NST+HT+FB', 1e-2, 1e-2, 1000),
                    NST_HT_var('NST+HT+subFB', 1e-2, 1e-2, 1000),
                    NST_HT_var('NST+stretchedHT', 1e-2, 1e-2, 1000),                        
                    Preconditioner(IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(OrthogonalMatchingPursuit(1e-7, algorithm='sparsify_QR'),                                rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(ApproximateMessagePassing(1e-7, 1000),                                                   rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(NST_HT_var('NST+HT+FB', 1e-2, 1e-2, 1000),                                               rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(NST_HT_var('NST+HT+subFB', 1e-2, 1e-2, 1000),                                            rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(NST_HT_var('NST+stretchedHT', 1e-2, 1e-2, 1000),                                         rcond=kwargs['rcond'], rinv=kwargs['rinv'])
                    ]
    p.solver_names = [get_solver_shortname(solver) for solver in p.solvers]
    return p


def make_params_randn_allsolvers_fast(*args, **kwargs):
    p = make_params_randn(*args, **kwargs)
    # Overwrite solvers
    p.solvers      = [
                    IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), 
                    IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), 
                    OrthogonalMatchingPursuit(1e-7, algorithm='sparsify_QR'),
                    ApproximateMessagePassing(1e-7, 1000), 
                    NST_HT_var('NST+HT+FB', 1e-2, 1e-2, 1000, max_num_data=20),
                    NST_HT_var('NST+HT+subFB', 1e-2, 1e-2, 1000),
                    NST_HT_var('NST+stretchedHT', 1e-2, 1e-2, 1000),                        
                    Preconditioner(IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(OrthogonalMatchingPursuit(1e-7, algorithm='sparsify_QR'),                                rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(ApproximateMessagePassing(1e-7, 1000),                                                   rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(NST_HT_var('NST+HT+FB', 1e-2, 1e-2, 1000, max_num_data=20),                                               rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(NST_HT_var('NST+HT+subFB', 1e-2, 1e-2, 1000),                                            rcond=kwargs['rcond'], rinv=kwargs['rinv']),
                    Preconditioner(NST_HT_var('NST+stretchedHT', 1e-2, 1e-2, 1000),                                         rcond=kwargs['rcond'], rinv=kwargs['rinv'])
                    ]
    p.solver_names = [get_solver_shortname(solver) for solver in p.solvers]
    return p


def get_solver_shortname(solver):
    if isinstance(solver, IterativeHardThresholding):
        if solver.mu == 0:
            name = 'IHTa'
        elif solver.mu == 1:
            name = 'IHTmu1'
        else:
            ValueError('Don\'t know name for this mu')
        
    elif isinstance(solver, OrthogonalMatchingPursuit):
        name = 'OMP'
    elif isinstance(solver, ApproximateMessagePassing):
        name = 'AMP'
    elif isinstance(solver, Preconditioner):
        name = 'P-' + get_solver_shortname(solver.solver)
    elif isinstance(solver, NST_HT_var):
        name = solver.variant
    elif isinstance(LeastSquaresIHT):
        if solver.mu == 0:
            name = 'IAPa'
        elif solver.mu == 1:
            name = 'IAPmu1'
        else:
            ValueError('Don\'t know name for this mu')
    else:
        ValueError('Unknown solver')

    return name
