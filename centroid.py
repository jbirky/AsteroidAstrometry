import numpy as np
import matplotlib.pyplot as plt
import math, os
import itertools


def findStar(red, **kwargs):
    """
    Find all points in a 2D image with ADU counts above a percentile threshold.
    """
    perc = kwargs.get('perc', 99)
    cut = np.percentile(red.flatten(), perc)
    
    pts = []
    for row in range(red.shape[0]):
        idx = np.where(red[row] >= cut)[0]
        if len(idx) != 0:
            for col in idx:
                pts.append([row, col])
    pts = np.array(pts)
    
    return pts


def dist(r1, r2):
    """
    Return Euclidean distance between two 2D vectors.
    """
    return np.sqrt((r1[0]-r2[0])**2 + (r1[1]-r2[1])**2)


def removePixels(pts, **kwargs):
    """
    Remove points associated with bad rows or columns of the detector.
    """
    rows = kwargs.get('rows', [])
    cols = kwargs.get('cols', [])
    
    ypts, xpts = [], []
    for p in pts:
        if (p[0] not in rows) & (p[1] not in cols):
            xpts.append(p[0])
            ypts.append(p[1])
    pts = np.array([ypts, xpts])
    
    return pts.T


def clusterPoints(points, **kwargs):
    """
    Cluster points into separate arrays if they are within the distance
    of a specified radius.
    """
    radius = kwargs.get('radius', 5)

    clusters = []
    while len(points) > 0:
        p0 = points[0]
        clus = []
        new_list = []
        for pt in points:
            if dist(p0, pt) < radius**2:
                clus.append(pt)
            else:
                new_list.append(pt)
        clusters.append(np.array(clus))
        points = new_list
    
    return np.array(clusters)


def findClusterMax(red, clusters):
    """
    Given clusters of bright points, find the pixel positions
    corresponding to max ADU count in each cluster.
    """
    peaks = []
    for clus in clusters:
        adu_vals = []
        for pt in clus:
            adu_vals.append(red[pt[0], pt[1]])
        max_adu_idx = np.where(adu_vals == max(adu_vals))[0][0]
        peaks.append(clus[max_adu_idx])
        
    return np.array(peaks)


def selectCircle(**kwargs):
    """
    Select a circle of discrete points centered around 'xc' and 'yc' within 
    a given radius size, and within the square dimensions of the image size.
    """
    xc, yc = kwargs.get('xc'), kwargs.get('yc')
    radius = kwargs.get('radius')
    dim = kwargs.get('dim', [np.inf, np.inf])
    
    ylist = np.arange(yc-radius, yc+radius+1)
    xlist = np.arange(xc-radius, xc+radius+1)
    square_cut = [list(xy) for xy in list(itertools.product(ylist, xlist))]

    circle_cut = []
    for pt in square_cut:
        if (dist([yc,xc], pt) < radius) & (0 <= pt[0] < dim[0]) & (0 <= pt[1] < dim[1]):
            circle_cut.append(pt)
            
    return circle_cut


def centroid(red, peaks, **kwargs):
    """
    Given peak pixels of stars, cut out a radius of pixels and
    compute sub-pixel centroids and errors
    """
    radius = kwargs.get('radius', 20)
    
    centroids = []
    for pt in peaks:
        circle = selectCircle(yc=pt[0], xc=pt[1], radius=radius, dim=red.shape)
        
        I, xI, yI = [], [], []
        for yx in circle:
            I.append(red[yx[0], yx[1]])
            yI.append(red[yx[0], yx[1]]*yx[0])
            xI.append(red[yx[0], yx[1]]*yx[1])
        ycent = sum(yI)/sum(I)
        xcent = sum(xI)/sum(I)
        centroids.append([ycent, xcent])
    
    return np.array(centroids)