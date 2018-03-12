import numpy as np
import matplotlib.pyplot as plt




# nm = 4
# npar = 3
# nxt = 22


def plot_translucent_point(point,m=None,xmin=0,xmax=55000,ax=None,alpha = 0.05,npar=4,nm=3,fmt = 'r-',**kwargs):
    # indicices [hardcoded]
    # sindex = range(nm * npar, nm * npar + nm)

    # get x
    x = np.linspace(xmin,xmax)

    # get "measurements"
    b = point[:nm * npar].reshape(nm, -1)
    xpm = np.polynomial.polynomial.polyval(x, np.array(b).T).T

    #graphics
    if ax is None:
        ax = plt.gca()
    if m is None:
        ax.plot(x,xpm,alpha=alpha,**kwargs)
    else:
        ax.plot(x,xpm[:,m],fmt,alpha=alpha,**kwargs)


def plot_translucent_sample(sample,alpha=None,**kwargs):
    if alpha is None:
        alpha = np.max((0.01,1.0/sample.shape[0]))
    for i in range(sample.shape[0]):
        plot_translucent_point(sample[i,:],alpha=alpha,**kwargs)