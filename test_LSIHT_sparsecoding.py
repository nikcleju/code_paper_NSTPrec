__author__ = 'Nic'

import numpy
import os
import scipy.io
import datetime
from collections import namedtuple

# Manually set here the path to the pyCSalgos package
# available at https://github.com/nikcleju/pyCSalgos
import sys, socket
hostname = socket.gethostname()
if hostname == 'caraiman':
    pyCSalgos_path = '/home/nic/code/pyCSalgos'
elif hostname == 'nclejupc':
    pyCSalgos_path = '/home/ncleju/code/pyCSalgos'
elif hostname == 'nclejupchp':
    pyCSalgos_path = '/home/ncleju/Work/code/pyCSalgos'
sys.path.append(pyCSalgos_path)

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

from LSIHT import LeastSquaresIHT
from pyCSalgos import IterativeHardThresholding
from pyCSalgos import OrthogonalMatchingPursuit
from pyCSalgos import ApproximateMessagePassing

from pyCSalgos import SynthesisSparseCoding

from parameters_SC import make_params_randn, make_params_ECG, plotdecays, get_solver_shortname, plot_options, \
                            make_params_ECG_onlyAMPP_partcond, make_params_ECG_onlyNSTHTFB, make_params_ECG_allsolvers, make_params_randn_allsolvers, make_params_randn_allsolvers_fast

def rerun(p):
    time_start = datetime.datetime.now()
    print(time_start.strftime("%Y-%m-%d-%H:%M:%S:%f") + " --- Started running %s..."%(p.name))

    # Output data file
    data_filename = os.path.join(p.save_folder, p.name + '_savedata')  # no extension here, will be added when saved
    file_prefix   = os.path.join(p.save_folder, p.name)

    sc = SynthesisSparseCoding(dictionary=p.dictionary, rhos=p.rhos, numdata=p.num_data, snr_db_sparse=p.snr_db_sparse, snr_db_signal=p.snr_db_signal, solvers=p.solvers)
    sc.loaddata(matfilename=data_filename+'.mat' )
    #sc.run(processes=1, random_state=123)
    #sc.savedata(data_filename)
    #sc.savedescription(file_prefix)
    #sc.plot(thresh=p.success_thresh, basename=file_prefix, legend=p.solver_names, saveexts=['png', 'pdf', 'pickle'])

    time_end = datetime.datetime.now()
    print(time_end.strftime("%Y-%m-%d-%H:%M:%S:%f") + " --- Ended. Elapsed: " + \
          str((time_end - time_start).seconds) + " seconds")

def replot(filename, specialname=None, ylim=(None, None)):
    time_start = datetime.datetime.now()
    print(time_start.strftime("%Y-%m-%d-%H:%M:%S:%f") + " --- Started replot on  %s..."%(filename))

    # Output data file
    basename = filename[:-13] + '_replot' + (('_'+specialname) if specialname is not None else '')  # [:-13] means remove final '_savedata.mat' part

    sc = SynthesisSparseCoding(dictionary=numpy.random.randn(10,20), ks=[1,2])              # Use any inputs here, the true data will be loader below
    sc.loaddata(picklefilename1=filename[:-4]+'.pickle', 
                picklefilename2=filename[:-4]+'_data.pickle')                               # Load saved data
    
    # Manually set legend and plot styles here. Can deduce them from solvers's class, but 
    solvernames = [get_solver_shortname(solver) for solver in sc.solvers]
    
    sc.plot(thresh=None, basename=basename, legend=solvernames, saveexts=['png', 'pdf', 'pickle'], plot_options=plot_options, rhomax=0.55, ylim=ylim)  # Plot

    time_end = datetime.datetime.now()
    print(time_end.strftime("%Y-%m-%d-%H:%M:%S:%f") + " --- Ended. Elapsed: " + \
          str((time_end - time_start).seconds) + " seconds")

def run(p):

    time_start = datetime.datetime.now()
    print(time_start.strftime("%Y-%m-%d-%H:%M:%S:%f") + " --- Started running %s..."%(p.name))

    # Output data file
    data_filename = os.path.join(p.save_folder, p.name + '_savedata')  # no extension here, will be added when saved
    file_prefix   = os.path.join(p.save_folder, p.name)

    sc = SynthesisSparseCoding(dictionary=p.dictionary, rhos=p.rhos, numdata=p.num_data, snr_db_sparse=p.snr_db_sparse, snr_db_signal=p.snr_db_signal, solvers=p.solvers)
    sc.run(processes=1, random_state=123)
    sc.savedata(data_filename)
    sc.savedescription(file_prefix)
    sc.plot(thresh=p.success_thresh, basename=file_prefix, legend=p.solver_names, saveexts=['png', 'pdf', 'pickle'], plot_options=p.plot_options, ylim=p.ylim)
    sc.plot_suppport_recovered(basename=file_prefix+'_supp', legend=p.solver_names, saveexts=['png', 'pdf', 'pickle'])

    time_end = datetime.datetime.now()
    print(time_end.strftime("%Y-%m-%d-%H:%M:%S:%f") + " --- Ended. Elapsed: " + \
          str((time_end - time_start).seconds) + " seconds")


# Main program starts here
if __name__ == "__main__":

    # For profiling
    #import cProfile
    #cProfile.run('run()', 'profile')

    # Seed randomly
    numpy.random.seed()

    # For plotting the spectra of the random dictionaries
    #plotdecays(length=500, As=[1, 1, 1, 1], RCconsts=[1.5, 4.5, 9, 12], filename='save/SparseCoding/SC_decays_500')

    #############################
    # RANDN dictionaries
    #############################

    num_data = 100

    ### Run with all 14 algorithms

    # 500x1000, low-noise recovery (80dB)
    run(make_params_randn_allsolvers(specialname='all', signal_size=500, dict_size=1000, RCconst=1.5,  snr_db_signal=80, snr_db_sparse=numpy.Inf, num_data=num_data, rhomax=0.55, rcond=0, rinv=None, ylim=(0, 0.7)))   # Fullcond
    run(make_params_randn_allsolvers(specialname='all', signal_size=500, dict_size=1000, RCconst=6,    snr_db_signal=80, snr_db_sparse=numpy.Inf, num_data=num_data, rhomax=0.55, rcond=0, rinv=None, ylim=(0, 0.7)))   # Fullcond
    run(make_params_randn_allsolvers(specialname='all', signal_size=500, dict_size=1000, RCconst=9,    snr_db_signal=80, snr_db_sparse=numpy.Inf, num_data=num_data, rhomax=0.55, rcond=0, rinv=None, ylim=(0, 0.7)))   # Fullcond
    ### TOO LONG DIDN'T RUN run(make_params_randn_allsolvers(specialname='all', signal_size=500, dict_size=1000, RCconst=10.5, snr_db_signal=80, snr_db_sparse=numpy.Inf, num_data=num_data, rhomax=0.55, rcond=0, rinv=None, ylim=(0, 0.7)))   # Fullcond
    #
    # Replot to adjust Y scale
    replot('save/SparseCoding/SC_all_randn_80_500x1000_1p500_savedata.mat', ylim=(0, 0.7))
    replot('save/SparseCoding/SC_all_randn_80_500x1000_6p000_savedata.mat', ylim=(0, 1.4))
    replot('save/SparseCoding/SC_all_randn_80_500x1000_9p000_savedata.mat', ylim=(0, 1.4))
    replot('save/SparseCoding/SC_all_randn_80_500x1000_6p000_savedata.mat', specialname='zoom', ylim=(0, 0.3))
    replot('save/SparseCoding/SC_all_randn_80_500x1000_9p000_savedata.mat', specialname='zoom', ylim=(0, 0.3))

    # 500x1000, 20db noise
    # Full conditioning
    run(make_params_randn_allsolvers(specialname='all_fullcond', signal_size=500, dict_size=1000, RCconst=1.5,  snr_db_signal=20, snr_db_sparse=20, num_data=num_data, rhomax=0.55, rcond=0.00, rinv=0.0))
    run(make_params_randn_allsolvers(specialname='all_fullcond', signal_size=500, dict_size=1000, RCconst=6,    snr_db_signal=20, snr_db_sparse=20, num_data=num_data, rhomax=0.55, rcond=0.00, rinv=0.0))
    run(make_params_randn_allsolvers(specialname='all_fullcond', signal_size=500, dict_size=1000, RCconst=9,    snr_db_signal=20, snr_db_sparse=20, num_data=num_data, rhomax=0.55, rcond=0.00, rinv=0.0))
    ### TOO LONG DIDN'T RUNrun(make_params_randn(specialname='fullcond', signal_size=500, dict_size=1000, RCconst=10.5, snr_db_signal=20, snr_db_sparse=20, num_data=num_data, rhomax=0.55, rcond=0.00, rinv=0.0))
    #
    # Partial conditioning
    run(make_params_randn_allsolvers(specialname='all_partcond', signal_size=500, dict_size=1000, RCconst=1.5,  snr_db_signal=20, snr_db_sparse=20, num_data=num_data, rhomax=0.55, rcond=0.01, rinv=0.1))
    run(make_params_randn_allsolvers_fast(specialname='all_partcond', signal_size=500, dict_size=1000, RCconst=6,    snr_db_signal=20, snr_db_sparse=20, num_data=num_data, rhomax=0.55, rcond=0.01, rinv=0.1, ylim=(0, 1.5)))
    run(make_params_randn_allsolvers_fast(specialname='all_partcond', signal_size=500, dict_size=1000, RCconst=9,    snr_db_signal=20, snr_db_sparse=20, num_data=num_data, rhomax=0.55, rcond=0.01, rinv=0.1, ylim=(0, 1.5)))
    ### TOO LONG run(make_params_randn(specialname='partcond', signal_size=500, dict_size=1000, RCconst=10.5, snr_db_signal=20, snr_db_sparse=20, num_data=100, rhomax=0.55, rcond=0.01, rinv=0.1, ylim=(0, 2.5)))
    
    
    
    #############################
    # ECG dictionaries
    #############################

    signal_size=512
    dict_size = 512
    num_data = 100

    ### All 14 algorithms
    # Low noise (80db)
    run(make_params_ECG_allsolvers(specialname='all_nocond',   signal_size=signal_size, dict_size=dict_size, snr_db_signal= 80,  snr_db_sparse=numpy.Inf, num_data=num_data, rhomax=0.3, rcond=0,    rinv=None))
    run(make_params_ECG_allsolvers(specialname='all_fullcond', signal_size=signal_size, dict_size=dict_size, snr_db_signal= 80,  snr_db_sparse=numpy.Inf, num_data=num_data, rhomax=0.3, rcond=0,    rinv=0))
    run(make_params_ECG_allsolvers(specialname='all_partcond', signal_size=signal_size, dict_size=dict_size, snr_db_signal= 80,  snr_db_sparse=numpy.Inf, num_data=num_data, rhomax=0.3, rcond=0.01, rinv=0.1, ylim=(0,1.5)))   

    # High noise (20db noise, meassurement and sparsity noise)
    run(make_params_ECG_allsolvers(specialname='all_nocond',    signal_size=signal_size, dict_size=dict_size, snr_db_signal= 20,  snr_db_sparse=20, num_data=num_data, rhomax=0.3, rcond=0,    rinv=None, ylim=(0,2)))
    run(make_params_ECG_allsolvers(specialname='all_fullcond',  signal_size=signal_size, dict_size=dict_size, snr_db_signal= 20,  snr_db_sparse=20, num_data=num_data, rhomax=0.3, rcond=0,    rinv=0,    ylim=(0,2)))
    run(make_params_ECG_allsolvers(specialname='all_subspace1', signal_size=signal_size, dict_size=dict_size, snr_db_signal= 20,  snr_db_sparse=20, num_data=num_data, rhomax=0.3, rcond=0.1,  rinv=0.1,  ylim=(0,2)))
    run(make_params_ECG_allsolvers(specialname='all_subspace2', signal_size=signal_size, dict_size=dict_size, snr_db_signal= 20,  snr_db_sparse=20, num_data=num_data, rhomax=0.3, rcond=0.01, rinv=0.01, ylim=(0,2)))
    run(make_params_ECG_allsolvers(specialname='all_partcond2', signal_size=signal_size, dict_size=dict_size, snr_db_signal= 20,  snr_db_sparse=20, num_data=num_data, rhomax=0.3, rcond=0.01, rinv=0.1,  ylim=(0,2)))
    
    # # Only AMP-P, illustrate partcond:
    # run(make_params_ECG_onlyAMPP_partcond(specialname='v2_onlyAMPP', signal_size=signal_size, dict_size=dict_size, snr_db_signal= 20,  snr_db_sparse=20, num_data=num_data, rhomax=0.3, rcond=0.01,  rinv=0.1,  ylim=(0,1)))
