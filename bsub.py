def bsub(x,over):

    import numpy as np

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
