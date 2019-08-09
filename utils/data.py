import numpy as np
from mpi4py import MPI
import matplotlib
import matplotlib.pyplot as plt
import pylab

def get_data(dataset, comm=MPI.COMM_WORLD):
    rank = comm.rank
    fig = None
    means_gt = None

    if dataset == 'BIRCH_random':
        #-- Parameters for BIRCH data set --
        # Variance
        sigma_sq_gt = 1.0
        # number of clusters
        C = 25
        # dimension
        dim = np.sqrt(C)*4*np.sqrt(2)
        # number of samples per cluster
        n_samples = 100
        # name (for output file name)
        dataset_name = dataset+'-'+str(sigma_sq_gt)+'-'+str(C)+'-'+str(n_samples)
        #-----------------------------------

        from itertools import product
        cov_gt = sigma_sq_gt*np.eye(2)
        import os.path
        import h5py
        # try:
        #     os.remove(dataset_name+'.h5')
        # except:
        #     pass
        if os.path.isfile(dataset_name+'.h5'):
            h5file = h5py.File(dataset_name+'.h5', 'r')
            means_gt = h5file['gt/means'].value
            X = h5file['train/data'].value
            Y = h5file['train/label'].value
        else:
            # h5file = h5py.File(dataset+'-'+str(1.0)+'-'+str(C)+'-'+str(n_samples)+'.h5', 'r')
            # means_gt = h5file['gt/means'].value
            # ground truth means
            means_gt = np.random.uniform(size=(C,2))*dim
            X = np.empty((0,2))
            Y = np.empty((0))
            for c in xrange(C):
                X = np.append(X,np.random.multivariate_normal(means_gt[c],cov_gt,n_samples),axis=0)
                Y = np.append(Y, c)
            h5file = h5py.File(dataset_name+'.h5', 'w')
            h5file.create_dataset('gt/means', data=means_gt)
            h5file.create_dataset('train/data', data=X)
            h5file.create_dataset('train/label', data=Y)
            h5file.close()
        Y = np.asarray([int(y) for y in h5file['train/label'].value for _ in xrange(n_samples)])
        if rank == 0:
            plt.figure(figsize=(10,10),dpi=80)
            plt.scatter(X[:,0],X[:,1],s=1,color='yellow')
            plt.scatter(means_gt[:,0],means_gt[:,1],s=50,marker='o',color='blue')
            pylab.ylim([-4*np.sqrt(2)/2.,(np.sqrt(C)+0.5)*4*np.sqrt(2)])
            pylab.xlim([-4*np.sqrt(2)/2.,(np.sqrt(C)+0.5)*4*np.sqrt(2)])
            ax = plt.gca()
            try:
                ax.set_facecolor('gainsboro')
            except:
                ax.set_axis_bgcolor('gainsboro')
            # plt.ion()
            # pylab.savefig(dataset_name+'.pdf')
            # plt.show()
        parameters = {
            'dataset':dataset,
            'D':2,
            'sigma_sq':sigma_sq_gt,
            'n_components_gt':C,
            'n_samples':n_samples,
            'dataset_name':dataset_name,
        }
        gt_values = {'means':means_gt, 'sigma_sq':sigma_sq_gt}

    elif dataset[:5] == 'BIRCH':
        #-- Parameters for BIRCH data set --
        # Variance
        sigma_sq_gt = 1.0
        # dimensions
        try:
            D = int(dataset.split('-')[0][5:])
        except:
            D = 2
        # number of clusters
        try:
            C = int(dataset.split('-')[1])
        except:
            C = 4096
        # number of datapoints per cluster
        try:
            n_samples = int(dataset.split('-')[2])
        except:
            n_samples = 100

        # number of clusters per dimension
        C_per_dim = np.round(C**(1./D)).astype(np.int32)
        # name (for output file name)
        dataset_name = dataset.split('-')[0]+'-'+str(C)+'-'+str(C_per_dim)+'-'+str(sigma_sq_gt)+'-'+str(n_samples)
        from itertools import product
        # ground truth means
        means_gt = np.asarray(list(product(xrange(C_per_dim), repeat=D)))*4*np.sqrt(2)
        cov_gt = sigma_sq_gt*np.eye(D)
        parameters = {
            'dataset':dataset,
            'D':D,
            'sigma_sq':sigma_sq_gt,
            'n_components_gt':C_per_dim**D,
            'n_samples':n_samples,
            'dataset_name':dataset_name,
        }
      #-----------------------------------

        import os.path
        import h5py
        if not os.path.isfile(dataset_name+'.h5'):
            if rank == 0:
                if dataset.split('-')[0][-7:] == 'coreset':
                    X = means_gt
                    Y = range(C)
                else:
                    X = np.empty((0,D))
                    Y = np.empty((0))
                    for c in xrange(C_per_dim**D):
                        X = np.append(X,np.random.multivariate_normal(means_gt[c],cov_gt,n_samples),axis=0)
                        Y = np.append(Y, c)
                h5file = h5py.File(dataset_name+'.h5', 'w')
                h5file.create_dataset('train/data', data=X)
                h5file.create_dataset('train/label', data=Y)
                h5file.close()
        comm.Barrier()
        h5file = h5py.File(dataset_name+'.h5', 'r')
        X = h5file['train/data'].value
        Y = np.asarray([int(y) for y in h5file['train/label'].value for _ in xrange(n_samples)])
        if rank == 0 and D>1:
            fig = plt.figure(figsize=(10,10),dpi=80)
            plt.scatter(X[:,0],X[:,1],s=1,color='yellow')
            pylab.ylim([-9,C_per_dim*4*np.sqrt(2)+9])
            pylab.xlim([-9,C_per_dim*4*np.sqrt(2)+9])
            ax = plt.gca()
            try:
                ax.set_facecolor('gainsboro')
            except:
                ax.set_axis_bgcolor('gainsboro')
            # plt.ion()
            # plt.show()
        gt_values = {'means':means_gt, 'sigma_sq':sigma_sq_gt}

    elif dataset == 'KDD':
        X = np.loadtxt('./datasets/KDD2004/bio_train.dat')[:,3:]
        Y = np.loadtxt('./datasets/KDD2004/bio_train.dat')[:,2]
        dataset_name = dataset
        parameters = {
            'dataset':dataset,
            'D':X.shape[1],
            'n_samples':X.shape[0],
            'dataset_name':dataset_name
        }
        gt_values = None

    elif dataset == 'KDD1999':
        # skiprows=2200000, nrows=200000,
        import pandas as pd
        data = pd.read_csv('./datasets/KDD1999/kddcup.data.corrected', header = None, dtype={0:np.int32,1:object,2:object,3:object,4:np.int32,5:np.int32,6:np.int32,7:np.int32,8:np.int32,9:np.int32,10:np.int32,11:np.int32,12:np.int32,13:np.int32,14:np.int32,15:np.int32,16:np.int32,17:np.int32,18:np.int32,19:np.int32,20:np.int32,21:np.int32,22:np.int32,23:np.int32,24:np.float64,25:np.float64,26:np.float64,27:np.float64,28:np.float64,29:np.float64,30:np.float64,31:np.int32,32:np.int32,33:np.float64,34:np.float64,35:np.float64,36:np.float64,37:np.float64,38:np.float64,39:np.float64,40:np.float64,41:object})
        data = pd.get_dummies(data)
        X = data.values[:,:-1]
        Y = data.values[:,-1]
        print X.shape
        asd
        # TODO
        parameters = {
            'dataset':dataset,
            'D':X.shape[1],
            'n_samples':X.shape[0],
            'dataset_name':dataset_name
        }

    elif dataset[:-1] == '20170602152918_':

        # Generate artificial data
        # Variance
        sigma_sq = 0.2
        # number of samples per cluster
        n_samples = 100000
        # number of clusters
        C = 7
        # name (for output file name)
        dataset_name = dataset+'-'+str(sigma_sq)+'-'+str(n_samples)
        from itertools import product
        # ground truth means
        if dataset[-1] == '2':
            means_gt = np.asarray([[-2,0],[-1,0],[0,1],[0,0],[0,-1],[1,0],[2,0]])
        elif dataset[-1] == '3':
            means_gt = np.asarray([[-2,0],[-1,0],[0,0.2],[0,0],[0,-0.2],[1,0],[2,0]])
        cov_gt = sigma_sq*np.eye(2)

        parameters = {
            'dataset':dataset,
            'D':2,
            'sigma_sq':sigma_sq,
            'n_components_gt':7,
            'n_samples':n_samples,
            'dataset_name':dataset_name,
        }
        #-----------------------------------
        import os.path
        import h5py
        if os.path.isfile(dataset_name+'.h5'):
            h5file = h5py.File(dataset_name+'.h5', 'r')
            X = h5file['train/data'].value
            Y = h5file['train/label'].value
            Z = h5file['train/density'].value
        else:
            X = np.empty((0,2))
            Y = np.empty((0))
            for c in xrange(7):
                X = np.append(X,np.random.multivariate_normal(means_gt[c],cov_gt,n_samples),axis=0)
                Y = np.append(Y, c)

            # Calculate the point density (from statistics)
            # from sklearn.neighbors.kde import KernelDensity
            # kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
            # Z = kde.score_samples(X)

            # Calculate the point density (from ground-truth)
            from scipy.stats import multivariate_normal
            Z = np.zeros((X.shape[0]))
            for c in xrange(7):
                Z += multivariate_normal.pdf(X, mean=means_gt[c], cov=cov_gt)/7.

            h5file = h5py.File(dataset_name+'.h5', 'w')
            h5file.create_dataset('train/data', data=X)
            h5file.create_dataset('train/label', data=Y)
            h5file.create_dataset('train/density', data=Z)
            h5file.close()
        if rank == 0:
            fig = plt.figure(figsize=(7,4),dpi=80)

            # # Sort the points by density, so that the densest points are plotted last
            # idx = Z.argsort().astype('int')
            # plt.scatter(X[idx,0],X[idx,1],c=-Z[idx], s=5, edgecolor='', cmap="jet_r")

            # contour plot
            delta = 0.025
            x = np.arange(-3.0, 3.0, delta)
            y = np.arange(-1.5, 1.5, delta)
            coords = np.asarray([[x_coord,y_coord] for x_coord in x for y_coord in y])

            Z_contour = np.zeros(coords.shape[0])
            from scipy.stats import multivariate_normal
            for c in xrange(7):
                Z_contour += multivariate_normal.pdf(coords, mean=means_gt[c], cov=cov_gt)/7.

            z = Z_contour.reshape((x.shape[0],y.shape[0])).T

            CS = plt.contour(x, y, z, 8)
            plt.clabel(CS, fontsize=8, inline=1)

            # plt.scatter(means_gt[:,0],means_gt[:,1],s=80,marker='o',color='gray')
            plt.scatter(means_gt[:,0],means_gt[:,1],s=50,marker='o',color='black', zorder=2)
            pylab.xlim([-3,3])
            pylab.ylim([-1.5,1.5])
            ax = plt.gca()
            try:
                ax.set_facecolor('white')
            except:
                ax.set_axis_bgcolor('white')
            # plt.ion()
            # plt.show()
        gt_values = {'means':means_gt, 'covariances':cov_gt}

    elif dataset == 'SUSY':
        dataset_name = dataset
        import os.path
        import h5py
        if os.path.isfile(dataset_name+'.h5'):
            h5file = h5py.File(dataset_name+'.h5', 'r')
            X = h5file['train/data'].value
            Y = h5file['train/label'].value
        else:
            data = np.genfromtxt('./datasets/SUSY/SUSY.csv', delimiter=",")
            X = data[:,1:]
            Y = data[:,0]
            h5file = h5py.File(dataset_name+'.h5', 'w')
            h5file.create_dataset('train/data', data=X)
            h5file.create_dataset('train/label', data=Y)
            h5file.close()
        gt_values = None
        parameters = {
            'dataset':dataset,
            'D':X.shape[1],
            'n_samples':X.shape[0],
            'dataset_name':dataset_name
        }

    elif dataset == 'SONG':
        data = np.genfromtxt('./datasets/SONG/YearPredictionMSD.txt', delimiter=",")
        X = data[:,1:]
        Y = data[:,0]
        dataset_name = dataset
        parameters = {
            'dataset':dataset,
            'D':X.shape[1],
            'n_samples':X.shape[0],
            'dataset_name':dataset_name
        }
        gt_values = None

    try:
        D = X.shape[1]
        cov_gt = np.copy(gt_values['covariances'])
        if cov_gt.shape == (D,D):
            gt_values['covariances'] = np.empty((C,D,D)) #dim(C,D,D)
            for c in xrange(C):
                gt_values['covariances'][c] = cov_gt
        elif cov_gt.shape == (D,):
            gt_values['covariances'] = np.empty((C,D,D)) #dim(C,D,D)
            for c in xrange(C):
                gt_values['covariances'][c] = np.eye(D) * cov_gt
    except:
        pass

    return X, Y, parameters, gt_values