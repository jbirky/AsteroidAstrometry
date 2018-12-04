import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
import math


def formatRADEC(ras, des):
	"""
	Format RA/DEC from strings to degrees
	"""
	ra0 = 15*(float(ras[0:2]) + float(ras[3:5])/60. + float(ras[6:])/3600.)
	dsgn = np.sign(float(des[0:3]))
	dec0 = float(des[0:3]) + dsgn*float(des[4:6])/60. + dsgn*float(des[7:])/3600.

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
    x0 = y0 = 512
    
    x = f_p*Xproj + x0
    y = f_p*Yproj + y0
    
    return x, y


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
    
    ax = np.dot(B, np.dot(c, xccd))
    ay = np.dot(B, np.dot(c, yccd))
    
    T = np.array([[f_p*ax[0], f_p*ax[1], ax[2]],
                  [f_p*ay[0], f_p*ay[1], ay[2]],
                  [0,         0,         1    ]])

    xy = np.array([xccd, yccd, np.ones(N)])
    Tinv = np.linalg.inv(T)
    
    Xccd = np.dot(Tinv, xy)[0]
    Yccd = np.dot(Tinv, xy)[1]
    
    return Xccd, Yccd