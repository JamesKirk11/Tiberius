import matplotlib.pyplot as plt
import numpy as np


# The exact edges of the orders are observation-dependent, meaning you may need to play around with an offest to correctly mask the orders

# The order edges are calibrated to LP 714-47b, so order_offset = 0 in this case
# order_offset = 0

# TOI-1726b
order_offset = 0

# HAT-P-26b
order_offset = -30

# HD260655c
order_offset = 0

# order_edges = {'order84': np.array([[219, 176], [290, 265]])+order_offset,\
# 'order83': np.array([[290, 265], [379, 349]])+order_offset,\
# 'order82': np.array([[379, 349], [465, 430]])+order_offset,\
# 'order81': np.array([[465, 430], [541, 532]])+order_offset,\
# 'order80': np.array([[541, 532], [628, 617]])+order_offset,\
# 'order79': np.array([[628, 617], [717, 722]])+order_offset,\
# 'order78': np.array([[717, 722], [804, 808]])+order_offset,\
# 'order77': np.array([[804, 808], [892, 909]])+order_offset,\
# 'order76': np.array([[892, 909], [978, 1007]])+order_offset,\
# 'order75': np.array([[978, 1007], [1075, 1109]])+order_offset,\
# 'order74': np.array([[1075, 1109], [1182, 1216]])+order_offset,\
# 'order73': np.array([[1182, 1216], [1284, 1329]])+order_offset,\
# 'order72': np.array([[1284, 1329], [1383, 1439]])+order_offset,\
# 'order71': np.array([[1383, 1439], [1497, 1558]])+order_offset,\
# 'order70': np.array([[1497, 1558], [1599, 1680]])+order_offset,\
# 'order69': np.array([[1599, 1680], [1711, 1791]])+order_offset,\
# 'order68': np.array([[1715, 1801], [1807, 1905]])+order_offset}

order_edges = {'order83': np.array([[253, 202], [324, 291]])+order_offset,\
 'order82': np.array([[324, 291], [413, 375]])+order_offset,\
 'order81': np.array([[413, 375], [499, 456]])+order_offset,\
 'order80': np.array([[499, 456], [575, 558]])+order_offset,\
 'order79': np.array([[575, 558], [662, 643]])+order_offset,\
 'order78': np.array([[662, 643], [751, 748]])+order_offset,\
 'order77': np.array([[751, 748], [838, 834]])+order_offset,\
 'order76': np.array([[838, 834], [926, 935]])+order_offset,\
 'order75': np.array([[926, 935], [1012, 1033]])+order_offset,\
 'order74': np.array([[1012, 1033], [1109, 1135]])+order_offset,\
 'order73': np.array([[1109, 1135], [1216, 1242]])+order_offset,\
 'order72': np.array([[1216, 1242], [1318, 1355]])+order_offset,\
 'order71': np.array([[1318, 1355], [1417, 1465]])+order_offset,\
 'order70': np.array([[1417, 1465], [1531, 1584]])+order_offset,\
 'order69': np.array([[1539, 1596], [1621, 1698]])+order_offset,\
 'order68': np.array([[1633, 1706], [1745, 1817]])+order_offset,\
 'order67': np.array([[1747, 1829], [1835, 1933]])+order_offset}

def mask_NIRSPEC_data(frame,order_number,verbose=False):
    """A function to mask all Keck/NIRSPEC orders other than the one if interest.

    Inputs:
    frame - the fits image
    order_edge - the left hand and right hand pixels (in non-rotated frame) that define the edge of the order. Pulled from the hard-coded order_edge dictionary
    verbose - True/False - plot the frame with the selected order highlighted

    Returns
    f_masked - the frame (same shape as inputted) with all but the order of interest masked with zeros"""

    masked_frame = np.zeros_like(frame)*np.nan
    left_edge = np.poly1d(np.polyfit([0,2048],order_edges[order_number][0],1))(np.arange(2048))
    right_edge = np.poly1d(np.polyfit([0,2048],order_edges[order_number][1],1))(np.arange(2048))

    if verbose:
        if verbose == -1:
            verbose = False

    for i in range(2048):
        col_left = int(np.floor(left_edge[i]))
        col_right = int(np.ceil(right_edge[i]))
        masked_frame[i][col_left:col_right] = frame[i][col_left:col_right]

    if verbose:
        plt.figure()
        plt.subplot(211)
        vmin,vmax = np.nanpercentile(masked_frame,[10,99])
        plt.imshow(frame,vmin=vmin,vmax=vmax,aspect="auto")
        for k in order_edges.keys():
            t = plt.text(order_edges[k].mean(),1048,"%s"%k,rotation=90,color='k',fontsize=12)
            t.set_bbox(dict(facecolor='red', alpha=1, edgecolor='red'))
        plt.title("%s"%order_number)
        plt.plot(left_edge,np.arange(2048),color='g')
        plt.plot(right_edge,np.arange(2048),color='g')
        plt.title("Frame before order masking")

        plt.subplot(212)
        # vmin,vmax = np.nanpercentile(masked_frame,[10,90])
        plt.imshow(masked_frame,vmin=vmin,vmax=vmax,aspect="auto")
        plt.title("Frame after masking")
        plt.plot(left_edge,np.arange(2048),color='g')
        plt.plot(right_edge,np.arange(2048),color='g')
        t = plt.text(order_edges[order_number].mean(),1048,"%s"%order_number,rotation=90,color='k',fontsize=12)
        t.set_bbox(dict(facecolor='red', alpha=1, edgecolor='red'))
        if verbose == -2:
            plt.show()
        if verbose > 0:
            plt.show(block=False)
            plt.pause(verbose)
            plt.close()

    return masked_frame


def get_guess_locations(order_number):
    trace_guess_locations = int(np.round(np.mean((order_edges[order_number][0][0],order_edges[order_number][1][1]))))
    trace_search_widths = int(abs(order_edges[order_number][0][0]-order_edges[order_number][1][0]))
    return np.array([trace_guess_locations]),np.array([trace_search_widths])
