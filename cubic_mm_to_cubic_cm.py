import config
import numpy as np
import pickle

data_filename = config.experiment_id+config.suffix+"_data"
model_truth_filename = config.experiment_id+config.suffix+"_truth"

with open(data_filename ,'rb') as f:
    dataraw = pickle.load(f)

with open(model_truth_filename ,'rb') as f:
    t = pickle.load(f)

dataraw=dataraw/1000

ofname = data_filename
with open(ofname,'wb') as f:
    pickle.dump(dataraw,f)

t['x']=t['x']/1000.

covm = t['covm']
si = np.eye(covm.shape[0])*0.001
covm = si.dot(covm.dot(si))
t['covm'] = covm

b = t['b']
b[:,0]=b[:,0]*0.001
b[:,2]=b[:,2]*1000.
t['b'] = b

otfname = model_truth_filename
with open(otfname,'wb') as f:
    pickle.dump(t,f)
