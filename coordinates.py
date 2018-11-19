import numpy as np
from numpy import cos, sin


def formatRADEC(ras, des):
    """
    Format RA/DEC from strings to degrees
    """
    ra0 = 15*(float(ras[0:2]) + float(ras[3:5])/60. + float(ras[6:])/3600.)
    dsgn = np.sign(float(des[0:3]))
    dec0 = float(des[0:3]) + dsgn*float(des[4:6])/60. + dsgn*float(des[7:])/3600.
    
    return ra0, dec0


def angleToProjected(**kwargs):
    ra0 = kwargs.get('ra0')
    dec0 = kwargs.get('dec0')
    ra = kwargs.get('ra')
    dec = kwargs.get('dec')
    
    X = -cos(dec)*sin(ra0-ra) / (cos(dec0)*cos(dec)*cos(ra-ra0) + sin(dec)*sin(dec0))
    Y = -sin(dec0)*cos(dec)*cos(ra-ra0) - sin(dec)*cos(dec0) / (cos(dec0)*cos(dec)*cos(ra-ra0) + sin(dec)*sin(dec0))
    
    return X, Y