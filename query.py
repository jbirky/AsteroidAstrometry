import urllib
import numpy as np
import pdb

def usno(radeg,decdeg,fovam,epoch):
    """
    get USNO B-1 stars centered at
    radeg and decdeg (J2000.0) in degrees
    centered in a square field of view (arc min) 
    Corrects for proper motion to current epoch
    """
    str1 = 'http://webviz.u-strasbg.fr/viz-bin/asu-tsv/?-source=USNO-B1'
    str2 = '&-c.ra={:4.6f}&-c.dec={:4.6f}&-c.bm={:4.7f}/{:4.7f}&-out.max=unlimited'.format(radeg,decdeg,fovam,fovam)
    weburl = str1+str2
    
    print('Calling Vizier', weburl)
    try: # Python 2
        with urllib.urlopen(weburl) as url:
            s = url.read()
    except: # Python 3
        with urllib.request.urlopen(weburl) as url:
            s = url.read()
    sl = s.splitlines()
    sl = sl[45:-1] # get rid of header - updated Oct 2013
    
    name = np.array([])
    rad = np.array([]) # RA in degrees
    ded = np.array([]) # DEC in degrees
    rmag = np.array([]) # rmage
    
    for k in sl:
        try:
            kw = k.split('\t')
        except:
            kw = k.decode('UTF-8').split('\t')
        ded0 = float(kw[2])
        pmrad = float(kw[6])/3600e3/np.cos(np.deg2rad(ded0)) # comvert from mas/yr to deg/year
        pmded = float(kw[7])/3600e3
        name = np.append(name,kw[0])
        rad = np.append(rad,float(kw[1]) + pmrad*(epoch-2000.0))
        ded = np.append(ded,float(kw[2]) + pmded*(epoch-2000.0))
        try: 
            rmag = np.append(rmag,float(kw[12]))
        except:
            rmag = np.append(rmag,np.nan)
    return name,rad,ded,rmag