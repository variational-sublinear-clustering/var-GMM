from __future__ import print_function
from __future__ import unicode_literals

import datetime
import errno
import h5py
import itertools
import json
import numpy as np
import os
import sys
import time
import h5py
import cProfile

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
from scipy import linalg

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab

from truncated_GMM import TruncatedGaussianMixture as GMM
from utils.data import get_data


#===== Set Model Parameters =====================
# default parameters
params = {
    'algorithm' : 'var-GMM-S',
    'C'         : 400,
    'Cprime'    : 5,
    'G'         : 5,
    'Niter'     : 25,
    'Ninit'     : 0,
    'dataset'   : 'BIRCH2-400',
    'VERBOSE'   : False,
}

# parameters given by user via command line
try:
    user_params = dict(arg.split('=',1) for arg in sys.argv[1:])
except:
    print('WARNING: Could not read user parameters properly. Reverting to default parameters.')
    user_params = {}
params.update(user_params)
params['C'] = int(params['C'])
params['Cprime'] = int(params['Cprime'])
params['G'] = int(params['G'])
params['Niter'] = int(params['Niter'])
params['Ninit'] = int(params['Ninit'])
params['VERBOSE'] = True if params['VERBOSE'] == 'True' else False

# define the outputs
if params['VERBOSE']:
    params['VERBOSE'] = {
        'll' : True,
        'fe' : True,
        'qe' : True,
        'cs' : True,
        'nd' : True,
        'np' : 5,
    }
else:
    params['VERBOSE'] = {
        'll' : False,
        'fe' : False,
        'qe' : False,
        'cs' : False,
        'nd' : True,
        'np' : np.inf,
    }

#===== Instantiate Model ========================
gmm = GMM(params)

#===== Load Data ================================
X, Y, data_params, gt_values = get_data(params['dataset'], comm)

#===== Initialize Output ========================
# calculate ground-truth scores
VERBOSE = params['VERBOSE']
loglikelihood_gt = gmm.loglikelihood(
    X,
    gt_values
) if VERBOSE['ll'] and gt_values['means'] is not None and gt_values['sigma_sq'] is not None else 'not available'
qe_gt = gmm.quantization_error(
    X,
    gt_values['means'],
) if VERBOSE['qe'] and gt_values['means'] is not None else 'not available'
purity_score_gt, NMI_score_gt, AMI_score_gt = gmm.clustering_scores(
    X,
    Y,
    gt_values['means'],
) if VERBOSE['cs'] and gt_values['means'] is not None else ('not available', 'not available', 'not available')

if rank==0:
    print('data set: ', params['dataset'])
    print('#samples: ', X.shape[0])
    print('#features: ', X.shape[1])
    print('algorithm: ', params['algorithm'])
    print('C: ', params['C'])
    print('Cprime: ', params['Cprime'])
    print('G: ', params['G'])
    print('#Iterations: ', params['Niter'])
    print('#Initial E-Step Iterations: ', params['Ninit'])

    # create files
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    while True:
        filename = './output/{}/{}/C{}_K{}_G{}_N{}-{}/{}'.format(
            data_params['dataset_name'],
            params['algorithm'],
            str(params['C']), str(params['Cprime']), str(params['G']),
            str(params['Niter']), str(params['Ninit']),
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            file_handle = os.open(filename+'_results.txt', flags)
        except OSError as e:
            if e.errno == errno.EEXIST:  # Failed as the file already exists.
                pass
            else:  # Something unexpected went wrong so reraise the exception.
                raise
        else:  # No exception, so the file must have been created successfully.
            if VERBOSE['ll'] : print('ground-truth LogLikelihood: ', loglikelihood_gt)
            if VERBOSE['qe'] : print('ground truth Q-error:', qe_gt)
            if VERBOSE['cs'] : print('ground truth Purity:', purity_score_gt)
            if VERBOSE['cs'] : print('ground truth NMI:', NMI_score_gt)
            if VERBOSE['cs'] : print('ground truth AMI:', AMI_score_gt)
            with open(filename+'_results.txt','w') as file:
                if VERBOSE['ll'] : file.write('#ground truth LogLikelihood: ' + str(loglikelihood_gt)+'\n')
                if VERBOSE['qe'] : file.write('#ground truth Q-error: ' + str(qe_gt)+'\n')
                if VERBOSE['cs'] : file.write('#ground truth Purity: ' + str(purity_score_gt)+'\n')
                if VERBOSE['cs'] : file.write('#ground truth NMI: ' + str(NMI_score_gt)+'\n')
                if VERBOSE['cs'] : file.write('#ground truth AMI: ' + str(AMI_score_gt)+'\n')
                outstr = ('{:'+str(int(np.log10(params['Niter'])+1))+'}\t{:15}\t{:15}\t{:15}\t{:8}\t{:8}\t{:8}\t{:8}\n').format(
                    'n', 'Free Energy', 'LogLikelihood', 
                    'Q-Error', 'Purity', 'NMI', 'AMI', '#D-Evals'
                )                
                file.write(outstr)
            json.dump([params,data_params], open(filename+"_parameters.txt",'w'))
            break
else:
    filename = None

#===== Fit Model ================================
gmm.fit(X, Y, filename=filename)


#===== Output Results ===========================
if rank==0:
    h5f = h5py.File(''.join((filename,'.hdf5')), 'w')
    h5f.create_dataset('means', data=gmm.means)
    h5f.create_dataset('sigma_sq', data=gmm.sigma_sq)
    h5f.close()