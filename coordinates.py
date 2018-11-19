import numpy as np
from numpy import cos, sin


def angleToProjected(**kwargs):
    ra0 = kwargs.get('ra0')
    dec0 = kwargs.get('dec0')
    ra = kwargs.get('ra')
    dec = kwargs.get('dec')
    
    X = -cos(dec)*sin(ra0-ra) / (cos(dec0)*cos(dec)*cos(ra-ra0) + sin(dec)*sin(dec0))
    Y = -sin(dec0)*cos(dec)*cos(ra-ra0) - sin(dec)*cos(dec0) / (cos(dec0)*cos(dec)*cos(ra-ra0) + sin(dec)*sin(dec0))
    
    return X, Y