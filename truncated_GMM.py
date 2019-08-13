from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from builtins import range

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import itertools
import numpy as np
from mpi4py import MPI
from scipy.stats import entropy
from scipy.stats import multivariate_normal
from scipy.stats import mode as statsmode
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics

import kmc2
from sklearn.cluster.k_means_ import _k_init

import matplotlib
import matplotlib.pyplot as plt
import pylab
from timeit import default_timer as timer

class TruncatedGaussianMixture(object):
    """ A truncated variational isotropic Gaussian Mixture Model.

    This model uses Expectation Truncation to reduce the computational
    complexity, to investigate the relationship between GMMs and the
    k-means algorithm and to build k-means-like algorithms that scale
    sublinearly in the number of clusters.
    It allows for different degrees of truncation (where Cprime = 1
    recovers the k-means algorithm for isotropic Gaussians, and
    Cprime = C recovers the standard EM algorithm for
    GMMs).
    """

    def __init__(self, params={}, comm=MPI.COMM_WORLD):
        """ Initialize class variables """

        #===== Set Model Parameters =============
        self.params = {
            'algorithm'     : 'var-GMM-S',
            'C'             : 400,
            'Cprime'        : 5,
            'G'             : 5,
            'Niter'         : 25,
            'Ninit'         : 0,
            'init_values'   : None,
            'VERBOSE'       : {'ll' : False,
                               'fe' : False,
                               'qe' : False,
                               'cs' : False,
                               'nd' : False,
                               'np' : np.inf 
                              },
        }
        self.params.update(params)

        self.n_iteration = 0
        self.comm = comm
        self.ndistevals = np.zeros((self.params['Niter']), dtype='int32')

        # the model parameters will be initialized when fitting the
        # model to data, as the initilializing process takes the given
        # data into account
        self.means = None
        self.sigma_sq = None
        self.K = None


    def __repr__(self):
        return f"TruncatedGaussianMixture({self.params})"


    def fit(self, X, y_true=None, filename=None):
        """ Fit model to data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to fit the model to.

        y_true : array-like, shape (n_samples, )
            The class labels to calculate the clustering scores (not
            used for fitting of the model itself)

        filename : string
            The path/folder/file to save your outputs.
        """

        comm = self.comm
        rank = comm.rank
        size = comm.size

        #===== Initialization ===================
        # initialize model parameters
        self._initialize(X)
        theta = {'means'    : self.means,
                 'sigma_sq' : self.sigma_sq}

        # allocate data points to threads
        my_n = [i for i in range(X.shape[0]) if i*size//X.shape[0] == rank]
        my_X = X[my_n]
        print('rank {} data shape: {}'.format(rank, my_X.shape))

        my_N, D = my_X.shape
        N = np.empty((1) , dtype='int32')
        comm.Allreduce(np.asarray(my_N, dtype='int32'), N, op=MPI.SUM)
        N = N[0]

        C, Cprime, G = self.params['C'], self.params['Cprime'], self.params['G']
        Niter, Ninit = self.params['Niter'], self.params['Ninit']
        algorithm = self.params['algorithm']
        VERBOSE = self.params['VERBOSE']

        # initialize K randomly
        self.K = np.asarray([np.random.choice(C, Cprime) for _ in range(my_N)]).astype(np.int32)

        # initialize G_c randomly, but making sure that c is in G_c
        if algorithm[:9] == 'var-GMM-S':
            self.G_c = np.asarray([
                np.concatenate(
                    [np.asarray([c]),
                     np.random.permutation(
                        np.delete(np.arange(C), np.asarray([c])))
                    ],axis=0)[:G]
                for c in range(C)])
        else:
            self.G_c = None

        # iterate Ninit times to gain better K and G_c
        if Cprime < C:
            for _ in range(Ninit):
                _, self.K, self.G_c = self._e_step(my_X, algorithm, Cprime)


        #===== Calculate and save initial clustering scores =====
        def verbose_output(my_X, y_true, theta, resp, filename):
            free_energy = self.free_energy(my_X, theta, resp, distributed=True) if VERBOSE['fe'] else None
            loglikelihood = self.loglikelihood(my_X, theta, distributed=True) if VERBOSE['ll'] else None
            q_error = self.quantization_error(my_X, theta['means'], distributed=True) if VERBOSE['qe'] else None
            purity_score, NMI_score, AMI_score = self.clustering_scores(my_X, y_true, theta['means'], distributed=True) if VERBOSE['cs'] else (None, None, None)
            ndistevals = np.empty((1) , dtype='int32')
            comm.Allreduce(self.ndistevals[self.n_iteration-1], ndistevals, op=MPI.SUM)
            if rank == 0:
                strn = ('{:'+str(int(np.log10(Niter)+1))+'}').format(self.n_iteration)
                strfe = '\t{:13.6f}'.format(free_energy) if VERBOSE['fe'] else '\t{:13}'.format('--')
                strll = '\t{:13.6f}'.format(loglikelihood) if VERBOSE['ll'] else '\t{:13}'.format('--')
                strqe = '\t{:13.6f}'.format(q_error) if VERBOSE['qe'] else '\t{:13}'.format('--')
                strpur = '\t{:8.6f}'.format(purity_score) if VERBOSE['cs'] else '\t{:8}'.format('--')
                strnmi = '\t{:8.6f}'.format(NMI_score) if VERBOSE['cs'] else '\t{:8}'.format('--')
                strami = '\t{:8.6f}'.format(AMI_score) if VERBOSE['cs'] else '\t{:8}'.format('--')
                strnd = '\t{}/{} (x{:.2f})'.format(ndistevals[0],N*C, N*C/ndistevals[0]) if VERBOSE['nd'] and self.n_iteration > 0 else '\t{:8}'.format('--')
                outstr = strn + strfe + strll + strqe + strpur + strnmi + strami + strnd
                strend = '\n'
                print(outstr, end=strend)
                if filename is not None:
                    with open(''.join((filename,'_results.txt')),'a') as file:
                        file.write(outstr+strend)

                if self.params['dataset'][0:5] == 'BIRCH' and (self.n_iteration%VERBOSE['np']==0 or self.n_iteration == Niter):
                    # plot BIRCH dataset and model means
                    fig = plt.figure(figsize=(10,10),dpi=80)
                    plt.scatter(X[:,0],X[:,1],s=1,color='yellow')
                    pylab.ylim([-9,np.sqrt(C)*4*np.sqrt(2)+9])
                    pylab.xlim([-9,np.sqrt(C)*4*np.sqrt(2)+9])
                    ax = plt.gca()
                    try:
                        ax.set_facecolor('gainsboro')
                    except:
                        ax.set_axis_bgcolor('gainsboro')
                    plt.scatter(theta['means'][:,0], theta['means'][:,1],
                                s=60,marker='x',color='#CC0000', zorder=3)
                    pylab.savefig(filename+'_'+str(self.n_iteration)+'.png')
                    plt.close(fig)

        # print headers
        if rank == 0:
            outstr = ('{:'+str(int(np.log10(Niter)+1))+'}\t{:13}\t{:13}\t{:13}\t{:8}\t{:8}\t{:8}\t{}').format(
                'n', 'Free Energy', 'LogLikelihood', 'Q-Error', 'Purity', 'NMI', 'AMI', '#D-Evals (Speed-Up)')
            print(outstr)

        # calculate initial scores (without updating model parameters and
        # therefore without counting the distance evaluations)
        if VERBOSE['fe'] or VERBOSE['ll'] or VERBOSE['qe'] or VERBOSE['nd'] or VERBOSE['cs']:
            stats, K, _ = self._e_step(my_X, algorithm, Cprime, countevals=False)
            tmp_theta = {'means' : theta['means']}
            tmp_theta['K'] = K
            resp = stats['posterior']
            # calculate sigma
            my_sigma_sq = 0.
            for r, x  in zip(resp, my_X):
                my_sigma_sq += np.sum(
                    r*np.square(np.linalg.norm(x-self.means,axis=1))
                    )
            sigma_sq = np.zeros(1, dtype='float64')
            comm.Allreduce(my_sigma_sq, sigma_sq, op=MPI.SUM)
            sigma_sq /= float(N*D)
            sigma_sq = sigma_sq[0]
            tmp_theta['sigma_sq'] = sigma_sq
            verbose_output(my_X, y_true, tmp_theta, resp, filename)

        #===== Learning iterations ==============
        self.training_time = 0.
        for self.n_iteration in range(1, Niter+1):
            self.start_time = timer()
            #--- E-step ---
            stats, theta['K'], G_c = self._e_step(my_X, algorithm, Cprime, G, countevals=VERBOSE['nd'])
            #--- M-step ---
            theta = self._m_step(my_X, stats, theta)
            self.training_time += timer() - self.start_time
            #--- output ---
            if self.n_iteration == Niter:
                VERBOSE['fe'], VERBOSE['ll'], VERBOSE['qe'], VERBOSE['cs'] = (True, True, True, True)
            if VERBOSE['fe'] or VERBOSE['ll'] or VERBOSE['qe'] or VERBOSE['cs'] or VERBOSE['nd']:
                verbose_output(my_X, y_true, theta, stats['posterior'], filename)
            #--- set parameters ---
            self.means, self.sigma_sq, self.K, self.G_c = theta['means'], theta['sigma_sq'], theta['K'], G_c        
        if rank == 0:
            print('Pure training time: {:.2f}s'.format(self.training_time))


    def _initialize(self, X):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to use for parameter initialization.
        """
        comm = self.comm
        rank = comm.rank
        size = comm.size
        if rank == 0:
            C = self.params['C']
            n_samples, D = X.shape
            self.means = kmc2.kmc2(X, C, afkmc2=True)
            self.sigma_sq = 1.
        self.means = comm.bcast(self.means, root=0)
        self.sigma_sq = comm.bcast(self.sigma_sq, root=0)


    def _e_step(self, X, algorithm=None, Cprime=None, G=None, countevals=False):
        """E step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to use for the E-Step.

        algorithm : string
            Choice of 'var-GMM-X' or 'var-GMM-S', with optional appended '+1'.
            Note: 'var-k-means-X' and 'var-k-means-S' are recoverd by
            using the corresponding 'var-GMM' algorithms with Cprime = 1

        Cprime : integer
            Truncation parameter. Number of non-zero values in the
            posterior distributions.

        G : integer
            Number of 'neighboring' clusters considered for each cluster
            in K to find new optimal K.

        countevals : booleans
            If True the number of distance evaluations are counted.

        Returns
        -------
        stats : dictionary {'posterior', 'log_joint_xc'}
            posterior distribution and the log-joint of x and c

        K : integer numpy array, shape (N samples, C' components)
            index set K(n)
        """
        comm = self.comm
        rank = comm.rank
        size = comm.size

        algorithm = self.params['algorithm'] if algorithm is None else algorithm
        assert algorithm in ['var-GMM-X', 'var-GMM-X+1', 'var-GMM-S', 'var-GMM-S+1',], \
            "algorithm must be 'var-GMM-X', 'var-GMM-X+1', 'var-GMM-S' or 'var-GMM-S+1'"
        add_random_c = algorithm[-2:] == '+1'
        algorithm = algorithm[:-2] if add_random_c else algorithm
        
        Cprime = self.params['Cprime'] if Cprime is None else Cprime
        G = self.params['G'] if G is None else G

        my_N, D = X.shape
        C = self.params['C']

        stats = {}

        def truncate(p, K, fill=-np.inf):
            """ Truncate distribution p to index set K

            Parameters
            ----------
            p : array-like, shape (N samples, C components)
                (probability) distribution to truncate

            K : integer array, shape (N samples, C' components)
                index set of non-zero elements

            fill: float
                fill value for values not in K, standard is either -inf
                for logarithmic distributions or 0 otherwise

            Returns
            -------
            truncated distribution : numpy array, shape (N samples, C components)
            """
            trunc_idx = self._idx_from_K(K)
            p_trunc = fill*np.ones_like(p, dtype=np.float64)
            for c in range(C):
                p_trunc[trunc_idx[c],c] = p[trunc_idx[c],c]
            return p_trunc


        def update_K(G_c, Cprime, add_random_c=False, countevals=False):
            """ Update variational truncation parameter K(n)

            Parameters
            ----------
            G_c : array-like, shape (C components, G components)
                index set of cluster neighbors

            Cprime : integer
                |K(n)| size of K(n)

            add_random_c : boolean
                True: add addional random cluster to G_n

            countevalse : boolean
                True: count number of evaluations

            Returns
            -------
            K : numpy array, shape (N samples, Cprime components)
                new index set K(n)

            G_n_log_joint_xc : numpy array, shape (N samples, C components)
                log-joint of x and c in G_n

            """

            # the union of neighbors of clusters in K(n) defines the
            # 'search space' G_n for closest clusters
            if add_random_c:
                G_n = [np.unique(np.append(np.concatenate(G_c[K_n]), np.random.choice(C))) for K_n in self.K]
            else:
                G_n = [np.unique(np.concatenate(G_c[K_n])) for K_n in self.K]

            # calculate log-joints
            G_n_log_joint_xc = self._log_joint_p_of_x_and_c(X, K=G_n, countevals=countevals)

            # find K to maximize the free energy based on the neighbors
            K = np.argpartition(G_n_log_joint_xc, C-Cprime, axis=1)[:,-Cprime:]

            return K, G_n_log_joint_xc


        if algorithm == 'var-GMM-X':
            #--- choose the neighbors of C as nearest neighboring
            #--- clusters by directly calculating the cluster distances

            # do C^2 distance evaluations
            # distances = np.zeros((C, C))
            # for i in range(C-1):
            #     distances[i,i+1:] = self._distance(self.means[i], self.means[i+1:,:], countevals=countevals)
            #     distances[i+1:,i] = distances[i,i+1:]
            distances = self._distance(self.means[:,np.newaxis,:], self.means[np.newaxis,:,:], countevals=countevals)

            # set neighbors G_c of clusters c by shortest distances
            G_c = np.argpartition(distances, G-1, axis=1)[:,:G]

            # update variational parameters K(n)
            K, G_n_log_joint_xc = update_K(G_c, Cprime, add_random_c, countevals)

            # truncate log-joint to the Cprime values in K(n)
            log_joint_xc = truncate(G_n_log_joint_xc, K, fill=-np.inf)

        elif algorithm == 'var-GMM-S':
            #--- choose the neighbors of cluster c based on the mean
            #--- responsibility of all data points belonging to the cluster

            # update variational parameters K(n)
            K, G_n_log_joint_xc = update_K(self.G_c, Cprime, add_random_c, countevals)

            # truncate log-joint to the Cprime values in K(n)
            log_joint_xc = truncate(G_n_log_joint_xc, K, fill=-np.inf)

            # derive new neighbors from the mean of all cluster data points
            cluster_datapoints = np.argmax(G_n_log_joint_xc, axis=1)
            G_c = np.empty((C, self.params['G'])).astype(np.int32)
            for c in range(C):
                cluster_distances = G_n_log_joint_xc[cluster_datapoints == c,:]

                # all_cluster_distances = np.concatenate(comm.allgather(cluster_distances))
                # the following is the buffered (more stable) version of
                # the above allgather command:
                NG = comm.allgather(cluster_distances.shape[0])
                displacements = np.cumsum([0]+NG[:-1])
                all_cluster_distances = np.zeros((np.sum(np.asarray(NG)),C),dtype=np.float64)
                comm.Barrier()
                comm.Allgatherv(
                    cluster_distances,
                    [all_cluster_distances,
                     np.asarray(NG)*C,
                     displacements*C,
                     MPI.DOUBLE])
                mask = ~np.isfinite(all_cluster_distances)
                masked_cluster_distances = np.ma.array(all_cluster_distances, mask=mask)
                mean_cluster_distance = masked_cluster_distances.mean(axis=0)
                mean_cluster_distance = np.ma.filled(mean_cluster_distance, -np.inf)
                mean_cluster_distance[c] = 0.
                G_c[c] = np.argpartition(mean_cluster_distance, C-G)[-G:]

        # calculate posteriors (aka 'responsibilities')
        resp = _softmax(log_joint_xc)
        stats = {'posterior' : resp, 'log_joint_xc' : log_joint_xc}
        return stats, K, G_c


    def _m_step(self, X, stats, theta):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data

        resp : array-like, shape (n_samples, n_components)
            Posterior probabilities (or responsibilities) of the point
            of each sample in X.

        Returns
        -------
        theta : dictionary {'means', 'sigma_sq'}
            Updated model parameters.
        """
        comm = self.comm
        rank = comm.rank

        my_N, D = X.shape
        N = np.empty((1) , dtype='int32')
        comm.Allreduce(np.asarray(my_N, dtype='int32'), N, op=MPI.SUM)
        C = self.params['C']
        log_joint_xc = stats['log_joint_xc']
        resp = stats['posterior']

        sum_resp = np.empty((C) , dtype='float64')
        comm.Allreduce( [np.sum(resp,axis=0), MPI.DOUBLE], [sum_resp, MPI.DOUBLE], op=MPI.SUM )

        #--- mu ---
        my_means = np.tensordot(resp, X, axes=(0,0))
        means = np.empty((C,D) , dtype=np.float64)
        comm.Allreduce( [my_means, MPI.DOUBLE], [means, MPI.DOUBLE], op=MPI.SUM )
        means[sum_resp!=0] /= sum_resp[sum_resp!=0, np.newaxis]
        theta['means'] = means

        #--- sigma_sq ---
        my_sum_X2 = np.zeros((C), dtype=np.float64)
        for r, x in zip(resp, X):
            X2_term = r*np.inner(x,x)
            # my_sum_X2, sum_error = SCS(my_sum_X2, X2_term + sum_error)
            my_sum_X2 += X2_term
        sum_X2 = np.empty((C) , dtype=np.float64)
        comm.Allreduce( [my_sum_X2, MPI.DOUBLE], [sum_X2, MPI.DOUBLE], op=MPI.SUM )
        Mu2 = np.asarray([np.inner(mean,mean) for mean in means])
        sigma_sq = np.sum(sum_X2 - Mu2*sum_resp)
        N = np.empty((1) , dtype='int32')
        comm.Allreduce(np.asarray(my_N, dtype='int32'), N, op=MPI.SUM)
        sigma_sq = sigma_sq/float(N*D)
        theta['sigma_sq'] = sigma_sq
        return theta


    def free_energy(self, X, theta=None, resp=None, distributed=False):
        """Free energy per data point

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input training samples.

        theta : dictionary
            Model parameters for free energy calculation.

        resp : array-like, shape (n_samples, n_components)
            Posterior probabilities (or responsibilities) of the point
            of each sample in X.

        distributed : boolean (default:True)
            Denote if the data is already distributed between processes or not.
            If not it is assumed, that each process holds the same data.

        Returns
        -------
        free_energy : float
        """

        rank = self.comm.rank
        size = self.comm.size

        if theta is None:
            theta = {'means':self.means,
                     'sigma_sq':self.sigma_sq,
                     'K':self.K}

        # split data between processes if not already distributed
        if not distributed:
            my_n = [i for i in range(X.shape[0]) if i*size//X.shape[0] == rank]
            my_X = X[my_n]
            if resp is not None:
                my_resp = resp[my_n]
            else:
                my_resp = None
        else:
            my_X = X
            my_resp = resp

        if self.params['Cprime'] == 1:
            free_energy = self._free_energy_GMM_isotropic_13(
                D=my_X.shape[1], sigma_sq=theta['sigma_sq'])
        else:
            free_energy = self._free_energy_GMM_isotropic_truncated_7(
                my_X=my_X, K=theta['K'], means=theta['means'],
                sigma_sq=theta['sigma_sq'])

        return free_energy


    def _free_energy_GMM_isotropic_truncated_7(self, my_X, K=None, means=None, sigma_sq=None):
        """Free energy of isotropic GMM (cov=sigma_sq*1) with \hat{\theta}=\theta
        LuckeForster2017 Eq.(7)

        Parameters
        ----------
        my_X : array-like, shape (n_samples, n_features)
            Distributed input training samples.

        K : array-like, shape (n_samples, Cprime)
            Index set of non-zero entries.

        means : array-like, shape (n_components, n_features)
            Model parameters for likelihood calculation.

        sigma_sq : float
            Common sigma^2 for isotropic GMMs.

        Returns
        -------
        free_energy : float
        """
        K = self.K if K is None else K
        means = self.means if means is None else means
        sigma_sq = self.sigma_sq if sigma_sq is None else sigma_sq
        C = self.params['C']
        my_N, D = my_X.shape
        comm = self.comm

        my_free_energy = 0.
        for n in range(my_N):
            my_free_energy_n = np.zeros((K[n].shape[0]))
            for i, c in enumerate(K[n]):
                my_free_energy_n[i] += multivariate_normal.logpdf(
                    my_X[n],
                    means[c],
                    sigma_sq*np.eye((D)),
                    allow_singular=True
                )
            shift = np.max(my_free_energy_n)-707.+np.log(C)
            my_free_energy += np.log(np.sum(np.exp(my_free_energy_n-shift)))+shift
        free_energy = np.zeros(1, dtype='float64')
        comm.Allreduce(my_free_energy, free_energy, op=MPI.SUM)
        N = np.empty((1) , dtype='int32')
        comm.Allreduce(np.asarray(my_N, dtype='int32'), N, op=MPI.SUM)
        free_energy = -np.log(C) + free_energy[0]/float(N)
        return free_energy


    def _free_energy_GMM_isotropic_truncated_18(self, resp, D, sigma_sq):
        """Free energy of isotropic GMM (cov=sigma_sq*1) with \hat{\theta}=\theta
        LuckeForster2017 Eq.(18)

        Parameters
        ----------
        resp : array-like, shape (n_samples, n_components)
            Posterior probabilities (or responsibilities) of the point
            of each sample in X.

        D : integer
            Data dimensionality.

        sigma_sq : float
            Common sigma^2 for isotropic GMMs.

        Returns
        -------
        free_energy : float
        """
        comm = self.comm
        C = self.params['C']

        my_N = resp.shape[0]
        N = np.empty((1) , dtype='int32')
        comm.Allreduce(np.asarray(my_N, dtype='int32'), N, op=MPI.SUM)

        my_sum_resp = np.sum((resp[resp>0]*np.log(resp[resp>0])))
        sum_resp = np.empty((1) , dtype='int32')
        comm.Allreduce(np.asarray(my_sum_resp, dtype='int32'), sum_resp, op=MPI.SUM)

        free_energy = -np.log(C) - float(D)/2.*np.log(2.*np.pi*np.e*sigma_sq) \
                      -1./float(N)*sum_resp[0]
        return free_energy


    def _free_energy_GMM_isotropic_13(self, D, sigma_sq=None):
        """Free energy of isotropic GMM (cov=sigma_sq*1) with
        \hat{\theta}=\theta and truncated with C'=1
        LuckeForster2017 Eq.(13)

        Parameters
        ----------
        D : integer
            Data dimensionality.

        sigma_sq : float
            Common sigma^2 for isotropic GMMs.

        Returns
        -------
        free_energy : float
        """
        sigma_sq = self.sigma_sq if sigma_sq is None else sigma_sq
        C = self.params['C']
        free_energy = -np.log(C) - float(D)/2.*np.log(2.*np.pi*np.e*sigma_sq)
        return free_energy


    def loglikelihood(self, X, theta=None, distributed=False):
        """Log-likelihood per data point

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input training samples.

        theta : dictionary
            Model parameters for likelihood calculation.

        sigma_sq : float
            Common sigma^2 for isotropic GMMs.

        distributed : boolean (default:False)
            Denote if the data is already distributed between processes or not.
            If not it is assumed, that each process holds the same data.

        Returns
        -------
        loglikelihood : float
        """
        rank = self.comm.rank
        size = self.comm.size

        theta = {'means' : self.means, 'sigma_sq' : self.sigma_sq} if theta is None else theta
        my_X = self._distribute(X) if not distributed else X

        loglikelihood = self.loglikelihood_GMM_isotropic(my_X, theta)
        return loglikelihood


    def loglikelihood_GMM_isotropic(self, my_X, theta):
        """Loglikelihood of isotropic GMM (cov=sigma_sq*1)

        Parameters
        ----------
        my_X : array-like, shape (n_samples, n_features)
            Distributed input training samples.

        means : array-like, shape (n_components, n_features)
            Gaussian means.

        sigma_sq : float
            Common sigma^2 for isotropic GMMs.

        Returns
        -------
        loglikelihood : float
        """
        theta = {'means' : self.means, 'sigma_sq' : self.sigma_sq} if theta is None else theta
        my_N, D = my_X.shape
        C = self.params['C']
        comm = self.comm

        N = np.empty((1) , dtype='int32')
        comm.Allreduce(np.asarray(my_N, dtype='int32'), N, op=MPI.SUM)
        my_loglikelihood = 0.
        for x in my_X:
            exp_arg = -1./(2.*theta['sigma_sq'])*np.square(np.linalg.norm(x-theta['means'], axis=1))
            shift = np.max(exp_arg)-707.+np.log(C)
            my_loglikelihood += np.log(np.sum(np.exp(exp_arg-shift)))+shift
        loglikelihood = np.zeros(1, dtype='float64')
        comm.Allreduce(my_loglikelihood, loglikelihood, op=MPI.SUM)
        loglikelihood = -np.log(C)-D/2.*np.log(2.*np.pi*theta['sigma_sq'])+1./float(N)*loglikelihood[0]
        return loglikelihood


    def quantization_error(self, X, means=None, distributed=False):
        """Quantization errorr

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input training samples.

        means : array-like, shape (n_components, n_features)
            Gaussian means.

        distributed : boolean (default:True)
            Denote if the data is already distributed between processes or not.
            If not it is assumed, that each process holds the same data.

        Returns
        -------
        q_error : float
        """
        means = self.means if means is None else means
        my_X = self._distribute(X) if not distributed else X

        my_closest_cluster = [np.argmin(np.linalg.norm(x - means, axis=1)) for x in my_X]
        my_q_error = np.sum(np.square(np.linalg.norm(means[my_closest_cluster] - my_X)))
        q_error = np.zeros((1) , dtype='float64')
        self.comm.Reduce(my_q_error, q_error, op=MPI.SUM, root=0)
        return q_error[0]


    def clustering_scores(self, X, y_true, means=None, distributed=False):
        """Purity, NMI and AMI scores

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input training samples.

        y_true : array-like, shape (n_samples, )
            Input training class labels.

        means : array-like, shape (n_components, n_features)
            Gaussian means.

        distributed : boolean (default:True)
            Denote if the data is already distributed between processes or not.
            If not it is assumed, that each process holds the same data.

        Returns
        -------
        purity_score : float
        NMI_score : float
        AMI_score : float
        """

        comm = self.comm
        means = self.means if means is None else means

        my_X = self._distribute(X) if not distributed else X
        my_N, D = my_X.shape

        # closest cluster to each X
        my_closest_cluster = [np.argmin(np.linalg.norm(x - means, axis=1)) for x in my_X]
        closest_cluster = np.concatenate(comm.allgather(my_closest_cluster))

        # classify cluster by most frequent labels
        cluster_labels = np.asarray([statsmode(y_true[np.where(closest_cluster==c)])[0][0] if c in closest_cluster else None for c in range(means.shape[0])])
        my_y_pred = cluster_labels[my_closest_cluster]
        y_pred = np.concatenate(comm.allgather(my_y_pred))

        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # purity
        purity_score = np.sum(np.amax(contingency_matrix, axis=0)) / float(np.sum(contingency_matrix))
        NMI_score = metrics.normalized_mutual_info_score(y_true, y_pred, average_method='warn')
        AMI_score = metrics.adjusted_mutual_info_score(y_true, y_pred, average_method='warn')
        return purity_score, NMI_score, AMI_score


    def _log_joint_p_of_x_and_c(self, my_X, means=None, sigma_sq=None, K=None, countevals=False):
        """Calculate p(x^{(n)},c|\theta)

        Parameters
        ----------
        my_X : array-like, shape (n_samples, n_features)
            Distributed input training data.

        means : array-like, shape (n_components, n_features)
            Gaussian means.

        sigma_sq : float
            Common sigma^2 for isotropic GMMs.

        K : array-like, shape (n_samples, Cprime)
            Index set of non-zero entries.

        countevalse : boolean
            True: count number of evaluations

        Returns
        -------
        log_joint_xc : float
        """
        means = self.means if means is None else means
        sigma_sq = self.sigma_sq if sigma_sq is None else sigma_sq
        K = self.K if K is None else K

        my_N = my_X.shape[0]
        C = self.params['C']

        log_joint_xc = (-np.inf)*np.ones((my_N, C), dtype=np.float64)
        for n, (k, x) in enumerate(zip(K, my_X)):
            log_joint_xc[n,k] = (-1./(2.*sigma_sq)*np.square(self._distance(x, means[k], countevals=countevals, distributed=True)))

        return log_joint_xc


    def _distance(self, X, means, countevals=False, distributed=True):
        """ calculate euclidean distance

        X : array-like, shape (n_samples, n_features)
            Input training data.

        means : array-like, shape (n_components, n_features)
            Gaussian means.

        distributed : boolean (default:True)
            Denote if the data is already distributed between processes or not.
            If not it is assumed, that each process holds the same data.
        """

        my_X = self._distribute(X) if not distributed else X

        distance = np.linalg.norm(my_X-means, axis=-1)

        if countevals:
            self.training_time += timer() - self.start_time
            my_N = my_X.shape[0] if len(my_X.shape) >= 2 else 1
            C = means.shape[-2] if len(means.shape) >= 2 else 1
            # print(my_N,C,my_N*C)
            # print(asd)
            self.ndistevals[self.n_iteration-1] += my_N*C
            self.start_time = timer()

        return distance


    def _distribute(self, X):
        """Distribute data between processes

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            input data

        Returns
        -------
        my_X : np-array, shape (~n_samples/n_processes, n_components)
            data for this process
        """
        rank = self.comm.rank
        size = self.comm.size
        my_n = [i for i in range(X.shape[0]) if i*size//X.shape[0] == rank]
        my_X = X[my_n]
        return my_X


    def _idx_from_K(self, K):
        """Index matrix to select the datapoints belonging to c out of K_n

        Parameters
        ----------
        K : array-like, shape (n_samples, Cprime)
            Index set of non-zero entries.

        Returns
        -------
        idx : numpy array, shape(C, n_samples)
        """
        N = K.shape[0]
        C = self.params['C']

        idx = np.zeros((C, N), dtype=bool)
        rows = np.arange(N).T
        idx.T[rows[:,np.newaxis], K] = True
        return idx


def _softmax(p):
    """Softmax function

    Parameters
    ----------
    p : array-like
        input distribution to softmax function

    Returns
    -------
    softmax : numpy array
    """
    maxes = np.amax(p, axis=1, keepdims=True)
    exp_p = np.exp(p - maxes)
    softmax = exp_p/np.sum(exp_p, axis=1, keepdims=True)
    return softmax


def SCS(a, b):
    """Singly compensated summation

    Parameters
    ----------
    a : float, integer, or array-like
        first summand

    b : float, integer, or array-like
        second summand

    Returns
    -------
    sum a + b
    summation error
    """
    s = a + b
    return s, b - (s - a)