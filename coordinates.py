import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import math
from centroid import dist


def formatRADEC(ras, des):
    """
    Format RA/DEC from strings to degrees
    """
    rasplit = ras.split(':')
    desplit = des.split(':')
    
    ra0 = 15*(float(rasplit[0]) + float(rasplit[1])/60. + float(rasplit[2])/3600.)
    dsgn = np.sign(float(desplit[0]))
    dec0 = float(desplit[0]) + dsgn*float(desplit[1])/60. + dsgn*float(desplit[2])/3600.

    return ra0, dec0


def angleToProjected(**kwargs):
    """
    Transform RA/DEC coordinates (degrees) into projected sky coordinates
    """
    ra0  = kwargs.get('ra0')
    dec0 = kwargs.get('dec0')
    ra   = kwargs.get('ra')
    dec  = kwargs.get('dec')
    fovam = kwargs.get('fovam', 4.0)

    #convert degrees to radians
    ra0  = ra0*math.pi/180
    dec0 = dec0*math.pi/180
    ra   = ra*math.pi/180
    dec  = dec*math.pi/180


    X = cos(dec)*sin(ra-ra0) / (cos(dec0)*cos(dec)*cos(ra-ra0) + sin(dec)*sin(dec0))
    Y = sin(dec0)*cos(dec)*cos(ra-ra0) - sin(dec)*cos(dec0) / (cos(dec0)*cos(dec)*cos(ra-ra0) + sin(dec)*sin(dec0))
    #convert to radians
    X, Y = -X, -Y

    if kwargs.get('plot', True) == True:
        plt.figure(figsize=[8,8])
        plt.scatter(X, Y, color='k', edgecolor='none')
        scale = fovam/2/60*math.pi/180 #convert arcmin -> arcsec -> radian
        plt.xlim(-scale, scale)
        plt.ylim(-scale, scale)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('USNO-B1.0 Projected Coordinates')
        plt.show()

    return X, Y


def projectedToPixelIdeal(Xproj, Yproj, **kwargs):
    """
    Apply ideal coordinate transformation.
    """
    f_p = kwargs.get('f_p', 190020)
    x0 = kwargs.get('x0', 512)
    y0 = kwargs.get('y0', 512)
    
    xpix = f_p*Xproj + x0
    ypix = f_p*Yproj + y0
    
    return xpix, ypix


def projectedToPixelIdealInv(Xccd, Yccd, **kwargs):
    """
    Apply ideal coordinate transformation.
    """
    f_p = kwargs.get('f_p', 190020)
    
    Xccd, Yccd = np.array(Xccd), np.array(Yccd)
    x0 = y0 = 512
    
    Xproj = (Xccd - x0)/f_p
    Yproj = (Yccd - y0)/f_p

    return Xproj, Yproj


def projectedToPixel(xccd, yccd, Xusno, Yusno, **kwargs):
    """
    Apply shear, scale and rotation coordinate transformation.
    """
    xccd, yccd   = np.array(xccd), np.array(yccd)
    Xusno, Yusno = np.array(Xusno), np.array(Yusno)
    
    f_p = kwargs.get('f_p', 190020)
    if len(xccd) != len(Xusno):
        print('xccd and Xusno are not the same dimension!')
        return
    else:
        N = len(xccd)
    
    B = np.array([f_p*Xusno, f_p*Yusno, np.ones(N)]).T
    c = np.dot(np.linalg.inv(np.dot(B.T, B)), B.T)
    
    ax = np.dot(c, xccd)
    ay = np.dot(c, yccd)
    
    #Get transformation matrix
    T = np.array([[f_p*ax[0], f_p*ax[1], ax[2]],
                  [f_p*ay[0], f_p*ay[1], ay[2]],
                  [0,         0,         1    ]])
    Tinv = np.linalg.inv(T)

    #Arrays of ccd coordinates which we want to transform
    xtrans = kwargs.get('xtrans', xccd)
    ytrans = kwargs.get('ytrans', yccd)
    xy = np.array([xtrans, ytrans, np.ones(len(xtrans))])
    
    #Convert coordinates using transformation matrix: X=Tinv*x
    Xccd = np.dot(Tinv, xy)[0]
    Yccd = np.dot(Tinv, xy)[1]

    return Xccd, Yccd


def projToSky(X,Y,alpha0,delta0):
    
    alpha0 = alpha0*math.pi/180
    delta0 = delta0*math.pi/180
    
    alpha = np.arctan(-(X)/(np.cos(delta0)-Y*np.sin(delta0)))+alpha0
    delta = np.arcsin((np.sin(delta0)+Y*np.cos(delta0))/(1+X**2+Y**2)**.5)
    
    alpha = alpha*180/math.pi
    delta = delta*180/math.pi
    
    return np.array([alpha,delta])


#====================
# Matching algorithms
#====================


def matchCoord(x, y, X, Y, **kwargs):

    N = len(x)
    M = len(X)
    
    img_size = kwargs.get('img_size', 1024)
    thres = kwargs.get('thres', 30)
    
    match_x, match_y = [], []
    match_X, match_Y = [], []
    for i in range(N):
        dist_i = []
        for j in range(M):
            dist_i.append(dist((x[i],y[i]), (X[j],Y[j])))
        closest = np.where(dist_i == min(dist_i))[0]
        if np.array(dist_i)[closest] < thres:
            match_x.append(x[i])
            match_y.append(y[i])
            match_X.append(X[closest][0])
            match_Y.append(Y[closest][0])
        
    return np.array(match_x), np.array(match_y), np.array(match_X), np.array(match_Y)


def removeMatches(x, y, xmatch, ymatch, **kwargs):
    
    if len(x) != len(xmatch):
        return
    else: N = len(x)
    
    img_size = kwargs.get('img_size', 1024)
    
    #Remove matches outside the image boundary
    xm1, ym1, xm2, ym2 = [], [], [], []
    for i in range(N):
        if (0 < xmatch[i] < img_size) & (0 < ymatch[i] < img_size):
            xm1.append(x[i])
            xm2.append(xmatch[i])
            ym1.append(y[i])
            ym2.append(ymatch[i])

    #Remove duplicate CCD points       
    xdict = {xm1[i]:xm2[i] for i in range(len(xm1))}
    ydict = {ym1[i]:ym2[i] for i in range(len(xm1))}
    
    xrev, yrev = {}, {}
    for k, v in xdict.items():
        xrev[v] = xrev.get(v, [])
        xrev[v].append(k)
    for k, v in ydict.items():
        yrev[v] = yrev.get(v, [])
        yrev[v].append(k)

    xc1, yc1, xc2, yc2 = [], [], [], []
    nset = len(xrev)
    for i in range(nset):
        #catalog
        key_x, key_y = xm2[i], ym2[i] 
        #ccd, list with multiples
        items_x, items_y = xrev[key_x], yrev[key_y] 
        nitems = len(items_x)
        
        dist_i = []
        for m in range(nitems):
            ccdm = [items_x[m], items_y[m]]
            catm = [key_x, key_y]
            dist_i.append(dist(ccdm, catm))
        close_idx = np.where(dist_i == min(dist_i))[0][0]
        #ccd 
        xc1.append(items_x[close_idx])
        yc1.append(items_y[close_idx])
        #catalog
        xc2.append(key_x)
        yc2.append(key_y)
            
    return xc1, yc1, xc2, yc2


def plotMatch(x, y, xmatch, ymatch, **kwargs):
    
    dim  = kwargs.get('dim', [0,1024])
    unit = kwargs.get('unit', 'pixel')
    lbl  = kwargs.get('lbl', ['x', 'y'])
    
    lines = [[(x[i], y[i]), (xmatch[i], ymatch[i])] for i in range(len(x))]
    lc = mc.LineCollection(lines, colors='k', linewidths=1, alpha=.5)
    fig, ax = plt.subplots(figsize=[8,8])
    ax.add_collection(lc)

    plt.scatter(x, y, color='b', edgecolor='none', label='CCD: %s'%(len(x)))
    plt.scatter(xmatch, ymatch, color='r', edgecolor='none', label='USNO: %s'%(len(xmatch)))
    plt.xlim(dim)
    plt.ylim(dim)
    plt.xlabel('%s [%s]'%(lbl[0], unit))
    plt.ylabel('%s [%s]'%(lbl[1], unit))
    plt.legend(loc='upper right', scatterpoints=1)
    plt.show()


def plotResiduals(x, y, X, Y, **kwargs):
    
    dim  = kwargs.get('dim', [0,1024])
    unit = kwargs.get('unit', 'pixel')
    lbl  = kwargs.get('lbl', ['x', 'y'])
    x, y = np.array(x), np.array(y)
    X, Y = np.array(X), np.array(Y)
    xerr = x - X
    yerr = y - Y
    
    fig, ax = plt.subplots(figsize=[12,6])
    plt.scatter(x, xerr, color='r', marker='^', label=lbl[0])
    plt.scatter(y, yerr, color='b', marker='v', label=lbl[1])
    plt.axhline(0, color='k', linestyle='--')
    plt.axhline(np.mean(xerr), color='r', linestyle='--')
    plt.axhline(np.mean(yerr), color='b', linestyle='--')

    plt.xlabel('%s or %s [%s]'%(lbl[0], lbl[1], unit))
    plt.ylabel('Error [%s]'%(unit))
    plt.legend(loc='upper left', scatterpoints=1)
    if 'title' in kwargs:
        plt.title(kwargs.get('title'))
    plt.xlim(kwargs.get('xlim', dim))
    yrngmax = max(abs(yerr))
    plt.ylim(kwargs.get('ylim', [-1.1*yrngmax, 1.1*yrngmax]))
    plt.show()