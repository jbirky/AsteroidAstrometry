import numpy as np
import astropy.io.fits as fits
from astropy.io import ascii
import numpy as np
import astropy.io.fits as fits
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('classic')
rc('font', **{'family': 'DejaVu Sans', 'serif': ['Computer Modern'], 'size':15})
rc('figure', facecolor='w')

def bsub(x,over):

    bias = np.mean(x[:,1024:1024+over],axis=1)

    # bias subtract
    for row in np.transpose(x):
        row = row - bias
        try:
            xbt = np.row_stack((xbt,row))
        except:
            xbt = row
           
    # Transpose the image back to original form
    xb = np.transpose(xbt)

    return xb[:,:-32]

def reduce(**kwargs):
    """
    Bias-subtract and normalize science images.
    """  
    dataf = kwargs.get('dataf')
    flatf = kwargs.get('flatf')
    
    flat = fits.getdata(flatf)
    dat = fits.getdata(dataf)
    hdr = fits.getheader(dataf)

    #Bias subtract
    datab = bsub(dat, hdr.get('cover')) 
    flatb = bsub(flat, hdr.get('cover')) 
    
    #Normalize
    flatb = flatb/np.median(flatb)
    reduced = datab/flatb
    
    return reduced

def plotImg(arr, **kwargs):
    """
    Plot 2D science image with identified objects.
    """
    avg = np.mean(arr.flatten())
    std = np.std(arr.flatten())
    dim = arr.shape
    rng = kwargs.get('rng', [np.percentile(arr, 5), np.percentile(arr, 95)])
    
    fig, ax = plt.subplots(1)
    pl = plt.imshow(arr, origin='lower', interpolation='nearest', cmap='gray_r', \
                    vmin=rng[0], vmax=rng[1])
    
    if 'pts' in kwargs:
        pts = kwargs.get('pts')
        plt.scatter(pts[0], pts[1], marker='o', s=50, facecolors='none', edgecolors='r')

    plt.colorbar(pl).set_label('Detector Value (ADU)')
    plt.xlabel('pixels(x)')
    plt.ylabel('pixels(y)')
    plt.xlim(0, dim[0])
    plt.ylim(0, dim[1])
    if 'title' in kwargs:
        plt.title(kwargs.get('title'))
    if 'save' in kwargs:
        plt.savefig(kwargs.get('save'))
    plt.show()