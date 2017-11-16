#IMPORTANT: suffix and experiment_id are imported from config.py:
import config # Sorryyyyyeyeyeyey :)
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano
import theano.tensor as tt
import pickle
# ## Init
# Init the env, load the generated data and true params
import os
plt.ion()
# identifier to be appended to filenames to enable multiple data
K = 3 #oplynomial degree+1
njobs = 6
nsamples = 18000
nsamples_aug = nsamples//njobs #how many samples to draw
nsamples = nsamples_aug*njobs
np.random.seed(1)
# set "environment"
# path_to_mcmc = '/home/madanh/.virtualenvs/transcendence/'
# experiment_dir = "/mnt/emptyplaceholder/mcmc/Experiments/20170428_0_boundary_b0/"
experiment_dir=os.getcwd()+'/'
experiment_id = config.experiment_id
suffix = config.suffix
# paths to generated data
data_file = os.path.join(experiment_dir, experiment_id + suffix + '_data')
# truth_file = os.path.join(experiment_dir, experiment_id + suffix + '_truth')

# The following code gives support for running notebooks both from pycharm and browser

curdir = os.getcwd()
print('We are in directory ' + curdir)


# Load the data
with open(data_file, 'rb') as f:
    data = pickle.load(f)

data = data #bring to cubic cm from cubic mm

M = data.shape[1]
N = data.shape[0]


## BET: FORMULATE THE MODEL
class TruncatedJeff(pm.Continuous):
    """
    Truncated Jeffreys prior for scale params
    """
    def __init__(self, *args, **kwargs):
        super( TruncatedJeff,self).__init__(*args,**kwargs)

    def logp(self,value):
        return tt.switch(value < 55.,
                         tt.switch(value > 0.001, -tt.log(value), -np.inf), -np.inf)

class Jeff(pm.Continuous):
    """
    Jeffreys prior for scale params
    """
    def __init__(self, *args, **kwargs):
        super(Jeff,self).__init__(*args,**kwargs)

    def logp(self,value):
         return -tt.log(value)

## BET: FORMULATE THE MODEL
mvg_model = pm.Model()
with mvg_model:
    # bias parametrization
    x = pm.Uniform('x',lower=0. ,upper=55., shape = (N,1))
    b0 = pm.Normal('b0',mu=0., sd=20., shape=M)
    b1 = pm.Normal('b1',mu=1., sd=0.5, shape=M)
    b2 = pm.Normal('b2',mu=0., sd=1./55., shape=M)
    xxx =pm.math.concatenate([x for _ in range(M)],axis = 1) #that list comprehension is just a "repmat"
    mu = xxx*xxx*b2+xxx*b1 + b0

    # covariance parametrization
    #sd_dist = TruncatedJeff.dist(shape=M)
    sd_template = pm.Bound(Jeff,lower = 0.001, upper = 55.)
    sd_dist = sd_template.dist(shape=M,testval=1.)
    chol_packed = pm.LKJCholeskyCov('chol_packed', n=M, eta = 1.,sd_dist=sd_dist)
    chol = pm.expand_packed_triangular(M,chol_packed)
    # data connection
    y = pm.MvNormal('y',mu = mu, chol= chol, shape = (N,M),observed = data)

## GIMEL: SAMPLE
with mvg_model:
    # step = pm.Slice()
    # map = pm.find_MAP()
    trace = pm.sample(nsamples_aug, njobs = njobs,tune=6000, target_accept=0.81)

## DALET: SAVE
# with open('trace','wb') as f:
#     pickle.dump(trace,f)

print("gelman-rubin:")
print(pm.gelman_rubin(trace))


## WAW: EXPORT IN A DIGESTABLE FORMAT
pos = np.zeros((nsamples,N+(M*(M+1))//2+K*M)) #prepare output placeholder
# format bias coeffs
for k in range(K):
    pos[:,k:K*M:K] = trace['b'+str(k)]
# define a funciton to format noise params
def chol_to_srs(in_chol):
    #recover the covariance matrix
    tril_indices = np.tril_indices(M)
    chol_e = np.zeros((M,M))
    chol_e[tril_indices[0], tril_indices[1]] = in_chol
    cov_e = np.dot(chol_e,chol_e.T)
    ## SRS decomposition of the result
    S = np.sqrt(np.diag(cov_e)) #only diagonal elements
    R = np.diag(1/S).dot(cov_e.dot(np.diag(1/S)))
    Rupper = R[np.triu_indices_from(R,1)]
    return np.concatenate((S,Rupper))
# apply the above funciton
for i in range(nsamples):
    pos[i,M*K:M*K+(M*(M+1))//2]=chol_to_srs(trace['chol_packed'][i,:])

# format the nuisance xpt
pos[:,-N:] = trace['x'].squeeze()

# serialize
storage_filename = experiment_dir+experiment_id+suffix+'trace.npy'
np.save(storage_filename ,pos)

# ## HE: VISUALIZE
# _ = pm.traceplot(trace)
# plt.show()
# plt.pause(0.001)
#
# input("Press Enter to continue...")
pass
