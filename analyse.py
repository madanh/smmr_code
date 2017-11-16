
# coding: utf-8
#IMPORTANT: suffix and experiment_id are imported from config.py:
import config # Sorryyyyyeyeyeyey :)
# ### Env and imports
import os
import sys
# import importlib, sys
import itertools
## numerical
import numpy as np
## graphics imports and setup
import matplotlib.pyplot as plt
import corner
### set nice style for graphs (white background instead of obnoxious beige)
# import seaborn as sb
# sb.set_style("ticks")
## io backends
import pickle

import reffreevis
####=================>>>>>>SWITCH<<<<<<=================####
enable_logging = False

switch_show = False
switch_abort_after_summary = False
if switch_show:
    plt.ion()
else:
    plt.ioff()
####=================>>>>>>><<<<<<<<<<<=================####
# set "environment"
# experiment_dir = "/mnt/emptyplaceholder/mcmc/Experiments/20170428_0_boundary_b0/"
experiment_dir=os.getcwd()+'/'
experiment_id = config.experiment_id

# **Please set the correct suffices below**
# identifier to be appended to filenames to enable multiple data
suffix = config.suffix

# ### Publication quality figures settings (you will not see it in the notebook, only in pdf)

# In[3]:

extra_tikz_axis_options = set(('axis x line = bottom',
                              'axis y line = left',
                              'axis equal')
                             )

#image_dir='/home/madanh/.virtualenvs/article_reffree_regression_for_qib/tex/images/bimodal/' #please terminate with /
image_dir = os.path.join(experiment_dir,'analysis/image'+experiment_id+suffix+'/')
try:
    os.makedirs(image_dir)
except os.error:
    print("image_dir already exists")



#logging infrastructure
if enable_logging:
    class Tee:
        def write(self, *args, **kwargs):
            self.out1.write(*args, **kwargs)
            self.out2.write(*args, **kwargs)
        def __init__(self, out1, out2):
            self.out1 = out1
            self.out2 = out2

    stdout_backup = sys.stdout
    sys.stdout = Tee(open(os.path.join(image_dir,"log.txt"), "w"), sys.stdout)

# ## Load experimental data
#dict storage for multiple experimentatl conditions
experiments = dict()
vis = True #turn visualization on/off
thinning = 1 #how many slices to skip (when threre's a lot of data numpy cant load all of it)
# paths to generated data
data_file = experiment_dir+experiment_id+suffix+'_data'

# Load the data
with open(data_file,'rb') as f:
    data = pickle.load(f)

#setup the labels for parameteres
nm = data.shape[1]
npar = 3
labels = ['b'+str(m)+str(k) for k in range(npar) for m in range(nm)] + \
         ['sigma'+str(m) for m in range(nm)] + \
         ['r'+str(i)+str(j) for i in range(nm) for j in range(i)] + \
         ['x'+str(p) for p in range(data.shape[0])]
# print(labels)
# load the sampler positions
storage_filename = experiment_dir+experiment_id+suffix+'trace.npy'
pos = np.load(storage_filename)

# Summary file for SMMR synth figures
inverse_scales = np.array(
    [1000., 1., 0.001, 1000., 1., 0.001, 1000., 1., 0.001, 1000., 1., 0.001, ]+
    [1000. for i in range(nm)]+
    [1. for i in range((nm*(nm-1))//2)]+
    [1000. for i in range(data.shape[0])]
)

summary=dict()
summary['m']=pos.mean(axis = (0))*inverse_scales
summary['h']=np.percentile(pos,95,axis = (0))*inverse_scales
summary['l']=np.percentile(pos,5,axis = (0))*inverse_scales

summary_file = experiment_dir+experiment_id+suffix+'_summary'

with open(summary_file,'wb') as f:
    pickle.dump(summary,f)

if switch_abort_after_summary:
    sys.exit(0)
## Reshape pos
nvar = pos.shape[-1]
# In[10]:
max_points = 2000
subsample = slice(0,pos.shape[0],max(pos.shape[0]//max_points,1))
# In[1]:



# In[17]:

rindex_start = (npar*nm+nm)
rindex_stop = ((npar*nm+nm)+nm*(nm-1)//2)
sindex_start=npar*nm
sindex_stop = (npar*nm+nm)
xindex_start = ((npar*nm+nm)+nm*(nm-1)//2)
def xpt_hist():
    """
    Show 1D histograms for each x_pt
    :return:
    """
    xindex = range(xindex_start,pos.shape[-1])
    a=int(np.ceil(np.sqrt(len(xindex))))  #side of the graph
    b=a  #int(np.ceil(a/0.75))
    fig,ax = plt.subplots(a,b,figsize=(12,9))
    for i in range(len(xindex)):

        ax.flat[i].hist(pos[:,xindex[i]])
        ax.flat[i].set_title("p="+str(i))
        plt.locator_params(nbins=2)
        # ax.flat[i].set_xticks((0,55000))

    plt.tight_layout()
    return

xpt_hist()
if switch_show:
    plt.show()
    plt.pause(0.001)
plt.savefig(os.path.join(image_dir,'xpt_hist.png'))


# =============== Save expected x_pt
xindex = slice(xindex_start,nvar)
m=pos[:,xindex].mean(axis = (0))
np.savetxt("expected_x.csv",m,delimiter = ',')

# =============== Make predictive curves
xmin = m.min()
xmax = m.max()
(fig,ax) = plt.subplots(1,nm,figsize =(10,7.5));
fmt_carousel = ['r-','k-','g-','b-','c-']
for m in range(nm):
    plt.sca(ax[m])
    reffreevis.plot_translucent_sample(pos[subsample,:],nm=nm,npar=npar,m=m,xmin=xmin,xmax=xmax,fmt=fmt_carousel[m])
    ax[m].plot([xmin,xmax],[xmin,xmax],'k--')
    ax[m].set_title("m="+str(m))
    plt.gca().set_aspect('equal')
plt.tight_layout()
if switch_show:
    plt.show()
    plt.pause(0.001)
plt.savefig(os.path.join(image_dir,'predicitve.png'),dpi=200)

## ===============TRUTH DEPENDENT VIS

truth_source_suffix = 'truth'
truth_file = experiment_dir+experiment_id+suffix+'_'+truth_source_suffix
# Load the truth
with open(truth_file,'rb') as f:
    truth = pickle.load(f)
x = truth['x']
if 'b' in truth:
    true_beta = np.array(truth['b'])
else:
    true_beta = np.array(truth['true_beta'])

true_covm = truth['covm']
S = np.sqrt(np.diag(true_covm)) #only diagonal elements
# S=np.diag(np.sqrt(np.diag(true_covm))) #full diagonal matrix
R = np.diag(1/S).dot(true_covm.dot(np.diag(1/S)))
Rupper = R[np.triu_indices_from(R,1)]

true_pos = np.concatenate((np.concatenate(true_beta),S,Rupper,x))
true_covm


# ### $E[\tilde{x}_{pt}]$ vs $x_{pt}$ x_pt
xindex = slice(xindex_start,pos.shape[-1])
fig, ax = plt.subplots(figsize = (7.5,7.5))
m=pos[:,xindex].mean(axis = (0))
h=np.percentile(pos[:,xindex],95,axis = (0))
l=np.percentile(pos[:,xindex],5,axis = (0))
ax.errorbar(x,m,[m-l,h-m],fmt ='ko')
ax.plot(np.array((0.,x.max())),np.array((0.,x.max())),'k--')
plt.gca().set_aspect('equal')
if switch_show:
    plt.show()
    plt.pause(0.001)
plt.savefig(os.path.join(image_dir,'xpt_versus.png'))
# sb.despine()
#plt.locator_params(nbins=5)

# =============== Make predictive curves
xmin = x.min()
xmax = x.max()
(fig,ax) = plt.subplots(1,nm,figsize =(10,7.5));
fmt_carousel = ['r-','k-','g-','b-','c-']
for m in range(nm):
    plt.sca(ax[m])
    reffreevis.plot_translucent_sample(pos[subsample,:],nm=nm,npar=npar,m=m,xmin=xmin,xmax=xmax,fmt=fmt_carousel[m])
    ax[m].plot([xmin,xmax],[xmin,xmax],'k--')
    ax[m].plot(x, data[:, m], 'ko', markersize=4)
    ax[m].plot(x, data[:, m], 'wo', markersize=3)
    ax[m].set_title("m="+str(m))
    plt.gca().set_aspect('equal')
plt.tight_layout()
if switch_show:
    plt.show()
    plt.pause(0.001)
plt.savefig(os.path.join(image_dir,'predicitve.png'),dpi=200)
# ### Corr and cov matrix

# In[24]:

r_e = pos[:,rindex_start:rindex_stop].mean(axis=(0))


# In[25]:

s_e = pos[:,sindex_start:sindex_stop].mean(axis=(0))


# In[26]:

R_e = np.eye(nm)*0.5
R_e[np.triu_indices_from(R_e,1)]=r_e
R_e = R_e + R_e.T


# In[27]:

covm_e = np.diag(s_e).dot(R_e.dot(np.diag(s_e)))
covm_e


# In[28]:

# In[29]:

fig,ax = plt.subplots(1,2)
ax[0].imshow(covm_e,interpolation='none',cmap = 'viridis')
#plt.colorbar()
ax[1].imshow(true_covm,interpolation='none',cmap='viridis')
#plt.colorbar()
if switch_show:
    plt.show()
    plt.pause(0.001)
plt.savefig(os.path.join(image_dir,'covariance_matrix.png'))


# In[30]:

inv_true_S = np.linalg.inv(np.diag(np.sqrt(np.diag(true_covm))))
true_R = inv_true_S.dot(true_covm.dot(inv_true_S))


# In[31]:

fig,ax = plt.subplots(1,2)
ax[0].imshow(R_e,interpolation='none',cmap = 'viridis')
#plt.colorbar()
ax[1].imshow(true_R,interpolation='none',cmap='viridis')
print(R_e, '\n\n', true_R)
if switch_show:
    plt.show()
    plt.pause(0.001)
plt.savefig(os.path.join(image_dir,'correlation_matrix.png'))


# ### Proper params for lulz

# In[32]:

b = true_beta


# In[33]:

# scale = (0.001,1,100000)
scale = (1,1,1)
bindex = np.prod(b.shape)
fnames = ['b0.tex','b1.tex','b2.tex']
for i in range(npar):
    fig, ax = plt.subplots()  #figsize = (1.56,1.56))
    true_params_range = np.array((true_pos[i:bindex:npar].min(),true_pos[i:bindex:npar].max()))*scale[i]
    ax.plot(true_params_range,true_params_range,'k--')
    m = pos[:,i:bindex:npar].mean(axis = (0))*scale[i]
    h = np.percentile(pos[:,i:bindex:npar],95,axis = (0))*scale[i]
    l = np.percentile(pos[:,i:bindex:npar],5,axis = (0))*scale[i]
    ax.errorbar(true_pos[i:bindex:npar]*scale[i],m,[h-m,m-l],fmt ='ko')
    #ax.minorticks_on()
    #ax.grid(True, which = 'both')
    plt.gca().set_aspect('equal')
    # sb.despine()
    plt.locator_params(nbins=3)
    
    # tikz_save(image_dir+fnames[i],extra = extra_tikz_axis_options,
    #      figurewidth = '\\figurewidth',
    #      figureheight = '\\figureheight',
    #          show_info = False)

    if switch_show:
        plt.show()
        plt.pause(0.001)
    plt.savefig(os.path.join(image_dir,fnames[i]+'.png'))

    
#ax.set_ylim((0.5,1.5))
'''
xs = true_pos[1:16:2]
ys = m
for j in range(len(xs)):
    plt.text(xs[j], ys[j], str(j), color="red", fontsize=bindex)
'''


# In[34]:

scale = 1
fig, ax = plt.subplots()#figsize = (1.56,1.56))
true_range = np.array((true_pos[sindex_start:sindex_stop].min(),true_pos[sindex_start:sindex_stop].max()))*scale
ax.plot(true_range,true_range,'k--')
m=pos[:,sindex_start:sindex_stop].mean(axis = (0))
h=np.percentile(pos[:,sindex_start:sindex_stop],95,axis = (0))
l=np.percentile(pos[:,sindex_start:sindex_stop],5,axis = (0))

ax.errorbar(true_pos[sindex_start:sindex_stop]*scale,m*scale,np.array([m-l,h-m])*scale,fmt ='ko')
#ax.grid(True, which = 'both')
plt.gca().set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.locator_params(nbins=3)
#ax.set_ylim((0.0,0.2))
#ax.set_xlim((0.0,0.2))
#ax.spines['left'].set_position('zero')
#ax.spines['bottom'].set_position('zero')
#ax.tick_params(direction='inout')
'''
xs = true_pos[sindex_start:sindex_stop]
ys = m
for j in range(len(xs)):
    plt.text(xs[j], ys[j], str(j), color="red", fontsize=12)

#ax.set_xticks(ax.get_xticks()[ax.get_xticks()!=0.0])
ax.set_yticks(
    ax.get_yticks()[
        ax.get_yticks()!=0.0])
        '''
print(h)
print(m)
print(l)

 
# tikz_save(image_dir+'s.tex',
#           extra = extra_tikz_axis_options,
#          figurewidth = '\\figurewidth',
#          figureheight = '\\figureheight',
#          show_info = False)
plt.title("$\sigma_m^{est}$ vs $\sigma_m^{"+truth_source_suffix+"}$")
if switch_show:
    plt.show()
    plt.pause(0.001)
plt.savefig(os.path.join(image_dir,'versus_sigma.png'))

# In[35]:

fig, ax = plt.subplots(figsize = (2,2))
tp = true_pos[rindex_start:rindex_stop]
true_range = (tp.min(),tp.max())
ax.plot(true_range,true_range,'k--')
m=pos[:,rindex_start:rindex_stop].mean(axis = (0))
h=np.percentile(pos[:,rindex_start:rindex_stop],95,axis = (0))
l=np.percentile(pos[:,rindex_start:rindex_stop],5,axis = (0))

ax.errorbar(tp,m,[m-l,h-m],fmt ='ko')
#ax.grid(True, which = 'both')
plt.gca().set_aspect('equal')

plt.locator_params(nbins=3)
# sb.despine()

if switch_show:
    plt.show()
    plt.pause(0.001)
plt.savefig(os.path.join(image_dir,'versus_corr.png'))

# In[36]:

for i in range(nm):
    # scales = [0.001,1,1000,0.001]
    scales = [1,1,1,1]
    indices = [npar*i+0,npar*i+1,npar*i+2,npar*nm+i]
    fig, axs = plt.subplots(4,4,figsize = (6.5,6.5))
    corner.corner(np.vstack((pos[subsample,indices[0]].flatten()*scales[0],
                             pos[subsample,indices[1]].flatten()*scales[1],
                             pos[subsample,indices[2]].flatten()*scales[2],
                             pos[subsample,indices[3]].flatten()*scales[3])).T,
                 labels = ['$b_0$','$b_1$','$b_2$','$\sigma$'],
                 truths = true_pos[indices]*scales,
                 max_n_ticks = 4,
                  truth_color='r',
                  levels=[0.25,0.5,0.75],
                  quantiles=[0.05,0.95],
                 fig = fig)
    pass
    if switch_show:
        plt.show()
        plt.pause(0.001)
    plt.savefig(os.path.join(image_dir,'corner_proper_'+str(i)+'.png'))


# In[37]:

corner.corner(pos[subsample,16:22].reshape(-1,22-16),
             truths = true_pos[16:22],
             truth_color='r',
             levels=[0.25,0.5,0.75],
             quantiles=[0.05,0.95],
             labels =['12','13','14','23','24','34'])
pass
if switch_show:
    plt.show()
    plt.pause(0.001)
plt.savefig(os.path.join(image_dir,'corner_corr.png'))


# In[38]:

corner.corner(pos[subsample,24:32].reshape(-1,32-24),
                           truths = true_pos[24:32])
pass
if switch_show:
    plt.show()
    plt.pause(0.001)
plt.savefig(os.path.join(image_dir,'corner_xpt.png'))

##===========+FUNCTIONS+===============



def xpt_truth():
    """
    print a human-readable table of xpt
    :return:
    """
    xindex = range(xindex_start,pos.shape[-1])
    print("p\ti\tx")
    for i in range(len(xindex)):
        print("%d\t%d\t%d"%(i,xindex[i],true_pos[xindex[i]]))

def xpt_scatter():
    """
    Show individual xpts vs true values as translucent markers to allow some insight into modes
    :return: 
    """
    xindex = range(xindex_start,pos.shape[-1])
    xindex = sorted(xindex, key=lambda i: true_pos[i])
    plt.subplots()
    plt.plot(true_pos[xindex],pos[subsample,xindex].squeeze().T,'r_',alpha=0.1)
    plt.plot(np.array((0.,x.max(axis=(0)))),np.array((0.,x.max(axis=(0)))),'k--')
    plt.gca().set_aspect('equal')


xpt_scatter()
if switch_show:
    plt.show()
    plt.pause(0.001)
plt.savefig(os.path.join(image_dir,'xpt_density_scatter.png'))

xpt_truth()




if enable_logging:
    sys.stdout = stdout_backup
if switch_show:
    input("Press Enter to continue...")

if switch_show:
    input("Press Enter to continue...")
