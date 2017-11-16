#IMPORTANT: suffix and experiment_id are imported from config.py:
import config # Sorryyyyyeyeyeyey :)
import pickle
import os
import numpy as np
# import click



# convenience function
def srs(covm):
    # decompose the covariance matrix into std and corr
    S = np.diag(np.sqrt(np.diag(covm)))
    inv_S = np.linalg.inv(S)
    R = inv_S.dot(covm.dot(inv_S))
    return (S,R)


def main(data_filename,model_truth_filename,output_filename="lse",degree = 2):
    """
    Create least square estimated of model parameter from generated data
    Model is polynomial bias, MVG random error

    Parameters
    ----------
    data_filename
    model_truth_filename
    output_filename
    :degree: degree of polynomial to use, defaults to 2

    Returns
    -------

    """
    # Load the data file
    with open(data_filename,'rb') as f:
        data = pickle.load(f)


    # Load model truth file
    with open(model_truth_filename,'rb') as f:
        model_truth = pickle.load(f)

    # Get xt
    x= model_truth["x"]

    # Fit the data to xt
    ## polynomial fitting as per http://stackoverflow.com/a/19165440/3791466
    b= [None]*data.shape[1]  # init the array with right number of els
    res= [None]*data.shape[1]
    for i in range(0, data.shape[1]):
        y = data[:, i]
        b[i], diag = np.polynomial.polynomial.polyfit(x, y, degree, full=True)
        # get residuals
        res[i] = np.polynomial.polynomial.polyval(x,b[i])-y

    # Get covm
    covm = np.cov(res,ddof=degree+1)

    # # Get corr (maybe)
    # S, R = srs(covm)
    # sigma = np.diag(S)

    # Generate an lse "truth" data

    # Serialize lse
    truth = {'x':x,'b':b,'covm':covm}
    with open(output_filename,'wb') as f:
        pickle.dump(truth,f)



if __name__ == "__main__":
    workdir = os.getcwd()+"/"
    experiment_id = config.experiment_id
    suffix = config.suffix
    data_filename = experiment_id+suffix+"_data"
    model_truth_filename = experiment_id+suffix+"_truth"
    output_filename= experiment_id+suffix+"_lse"
    main(os.path.join(workdir,data_filename),
         os.path.join(workdir,model_truth_filename),
         output_filename=os.path.join(workdir,output_filename)
         )

