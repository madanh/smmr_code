import config
import numpy as np
import pickle

data_filename = config.experiment_id+config.suffix+"_data"
model_truth_filename = config.experiment_id+config.suffix+"_truth"
lse_filename = config.experiment_id+config.suffix+"_lse"

with open(data_filename ,'rb') as f:
    dataraw = pickle.load(f)

dataraw=dataraw*1000

ofname = data_filename+'_mm'
with open(ofname,'wb') as f:
    pickle.dump(dataraw,f)

#Truth file
with open(model_truth_filename ,'rb') as f:
    t = pickle.load(f)

t['x']=t['x']*1000.

covm = t['covm']
si = np.eye(covm.shape[0])/0.001
covm = si.dot(covm.dot(si))
t['covm'] = covm

b = t['b']
b[:,0]=b[:,0]/0.001
b[:,2]=b[:,2]/1000.
t['b'] = b

otfname = model_truth_filename+'_mm'
with open(otfname,'wb') as f:
    pickle.dump(t,f)

#lse_file
with open(lse_filename ,'rb') as f:
    t = pickle.load(f)

t['x']=t['x']*1000.

covm = t['covm']
si = np.eye(covm.shape[0])/0.001
covm = si.dot(covm.dot(si))
t['covm'] = covm

b = np.array(t['b'])
b[:,0]=b[:,0]/0.001
b[:,2]=b[:,2]/1000.
t['b'] = b

otfname = lse_filename+'_mm'
with open(otfname,'wb') as f:
    pickle.dump(t,f)
