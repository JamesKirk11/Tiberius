import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, medfilt
import pickle
from astropy.io import fits

parser = argparse.ArgumentParser(description='Load all fits files within a directory and produced lists of file types')
# parser.add_argument('--quartz_flats','-q', help="""Enter list of quartz flats file names""")
# parser.add_argument('--milky_flats','-m', help="""Enter list of milky flats file names""")
# parser.add_argument('--arcs','-a', help="""Enter list of arcs file names""")
parser.add_argument('-c','--clobber',help="""Need this argument to save resulting fits file, default = False""",action='store_true')
parser.add_argument('-v','--verbose',help="""Display the image of each frame before combining it.""",action='store_true')
args = parser.parse_args()


def parse_sections(section_type,frame):

    # print(frame)

    header = frame[0].header
    data = frame[0].data

    # remove brackets and split
    row_part, col_part = header[section_type].strip('[]').split(',')

    # convert to slice objects
    r_start, r_end = map(int, row_part.split(':'))
    c_start, c_end = map(int, col_part.split(':'))

    row_slice = slice(r_start, r_end)
    col_slice = slice(c_start, c_end)

    return data[col_slice,row_slice].astype(float)


def median_combine(file_list,image_type="",verbose=False):

    bias_data = []
    science_data = []

    print("\n### Median combining %s ###\n"%file_list)

    file_names = np.loadtxt(file_list,dtype=str)

    if verbose:
        plt.figure()

    for f in file_names:

        print(f)

        fits_file = fits.open(f)

        bias = parse_sections("BIASSEC",fits_file)
        science = parse_sections("DATASEC",fits_file)
        science -= np.median(bias)

        fits_file.close()

        bias_data.append(np.median(bias))
        science_data.append(science)

        if verbose:

            vmin,vmax = np.nanpercentile(science,[10,90])
            plt.imshow(science,vmin=vmin,vmax=vmax,cmap='hot')
            plt.xlabel("X pixel")
            plt.ylabel("Y pixel")
            plt.title("%s: %s"%(image_type,f))
            plt.colorbar()
            plt.show(block=False)
            plt.pause(0.5)
            plt.clf()

    plt.figure()
    plt.plot(np.array(bias_data),'k.')
    plt.title(image_type)
    plt.ylabel("Bias median")
    plt.xlabel("Frame number")
    plt.show()

    return np.median(science_data,axis=0)




def get_order(row,prominence=0.01,distance=10):

    row_smooth = gaussian_filter1d(row, sigma=2)
    row_norm = row_smooth / (np.median(row_smooth) + 1e-8)
    # invert signal to find troughs
    troughs, _ = find_peaks(-row_norm, prominence=prominence, distance=distance)
    return troughs


def track_orders(image,max_jump_low=30, max_jump_high=10, verbose=True, prominence=0.01, distance=10):

    nrows,ncols = image.shape

    ref_row_idx = nrows//2

    # build reference array to account for the fact that the reference rows may find different numbers of orders
    arrays = [get_order(row,prominence,distance) for row in image[ref_row_idx-10:ref_row_idx+10]]
    # Step 1: build reference (e.g., from longest array)
    ref = max(arrays, key=len)

    aligned = []

    for a in arrays:
        aligned_arr = np.full(len(ref), np.nan)

        for val in a:
            # find closest index in reference
            idx = np.argmin(np.abs(ref - val))

            # only assign if "close enough"
            if abs(ref[idx] - val) <= 5:
                aligned_arr[idx] = val

        aligned.append(aligned_arr)

    aligned = np.array(aligned)
    ref_positions = np.nanmean(aligned, axis=0)

    # ref_positions = np.mean([get_order(row) for row in image[ref_row_idx-10:ref_row_idx+10]],axis=0).astype(int)

    nrows = image.shape[0]
    ntracks = len(ref_positions)

    # store results: one array per track
    tracks = [np.full(nrows, np.nan) for _ in range(ntracks)]

    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ref_positions,[nrows//2]*len(ref_positions),"kx",zorder=100000)
        ax.imshow(image)

    # initialize at reference row
    for i, x in enumerate(ref_positions):
        tracks[i][ref_row_idx] = x

    # --- track upward ---
    for y in range(ref_row_idx - 1, -1, -1):

        troughs = get_order(image[y])

        if verbose:
            ax.plot(troughs,[y]*len(troughs),'r.')

        for i in range(ntracks):
            prev = tracks[i][y + 1]
            if np.isnan(prev):
                continue

            if len(troughs) == 0:
                continue

            diffs = troughs - prev

            # asymmetric window: prefer movement to smaller x
            valid = troughs[
                (diffs >= -max_jump_low) & (diffs <= max_jump_high)
            ]

            if len(valid) == 0:
                # tracks[i][y] = prev
                continue  # track disappears

            else:
                # pick closest to previous position
                tracks[i][y] = valid[np.argmin(np.abs(valid - prev))]

    # --- track downward ---
    for y in range(ref_row_idx + 1, nrows):
        troughs = get_order(image[y])

        if verbose:
            ax.plot(troughs,[y]*len(troughs),'r.')

        for i in range(ntracks):
            prev = tracks[i][y - 1]
            if np.isnan(prev):
                continue

            if len(troughs) == 0:
                continue

            diffs = troughs - prev

            valid = troughs[
                (diffs >= -max_jump_low) & (diffs <= max_jump_high)
            ]

            if len(valid) == 0:
                # tracks[i][y] = prev
                continue

            else:
                tracks[i][y] = valid[np.argmin(np.abs(valid - prev))]

    if verbose:
        for tr in tracks:
            y = np.arange(len(tr))
            mask = ~np.isnan(tr)
            ax.plot(tr[mask], y[mask], lw=2)

        ax.set_title("Located tracks (colours) vs located troughs (red points)")
        fig.show()

    return np.array(tracks)


def smooth_tracks(image,tracks,median_box_width=11,sigma_clip=4,poly_order=4):

    nrows,ncols = image.shape

    smoothed_tracks = []
    for t in tracks:
        running_median = medfilt(t,median_box_width)
        finite = np.isfinite(running_median)

        if len(np.where(finite)[0]) < nrows*0.75:
            used_poly_order = 2
        else:
            used_poly_order = poly_order

        poly1 = np.poly1d(np.polyfit(np.arange(nrows)[finite],running_median[finite],used_poly_order))
        residuals = running_median[finite] - poly1(np.arange(nrows)[finite])
        keep_idx = ((residuals >= -sigma_clip*np.std(residuals)) & (residuals <= sigma_clip*np.std(residuals)))
        poly2 = np.poly1d(np.polyfit(np.arange(nrows)[finite][keep_idx],running_median[finite][keep_idx],used_poly_order))

        smooth_t = poly2(np.arange(nrows))
        smooth_t[smooth_t < 0] = 0
        smoothed_tracks.append(np.round(smooth_t).astype(int))

        # plt.figure()
        # plt.plot(running_median,np.arange(nrows),'r.')
        # plt.plot(running_median[finite][keep_idx],np.arange(nrows)[finite][keep_idx],'k.')
        # plt.plot(poly2(np.arange(nrows)),np.arange(nrows))
        # plt.show()

    plt.figure()
    # vmin,vmax = np.nanpercentile(image,[0.1,0.9])

    plt.imshow(image, origin='lower')#,vmin=vmin,vmax=vmax)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.title("Smoothed order tracks")

    for tr in smoothed_tracks:
        y = np.arange(len(tr))
        mask = ~np.isnan(tr)
        plt.plot(tr[mask], y[mask], lw=3)

    plt.show()

    return smoothed_tracks


def generate_order_mask(smoothed_tracks,image):

    ny, nx = image.shape
    y = np.arange(ny)[:, None]      # (ny, 1)
    x = np.arange(nx)[None, :]      # (1, nx)

    order_masks = {}

    for i in range(len(smoothed_tracks) - 1):
        x1 = smoothed_tracks[i][:, None]      # (ny, 1)
        x2 = smoothed_tracks[i+1][:, None]    # (ny, 1)

        # lower/upper edges per row
        xmin = np.minimum(x1, x2)
        xmax = np.maximum(x1, x2)

        mask = (x >= xmin) & (x < xmax)

        # remove rows where either edge is NaN
        valid = ~np.isnan(x1) & ~np.isnan(x2)
        mask &= valid

        order_masks["order%s"%(i+1)] = mask

    return order_masks


def save_fits(data,filename,clobber=False):
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(filename,overwrite=clobber)
    return


def main():

    # Blue quartz
    master_quartz_blue = median_combine("MIKE-Blue_Quartz_list.txt","Quartz flat - Blue",args.verbose)
    rough_tracks_blue = track_orders(master_quartz_blue,verbose=args.verbose)
    smoothed_tracks_blue = smooth_tracks(master_quartz_blue,rough_tracks_blue)
    order_masks_blue = generate_order_mask(smoothed_tracks_blue,master_quartz_blue)
    pickle.dump(order_masks_blue,open("MIKE-Blue_order_masks.pickle","wb"))
    save_fits(master_quartz_blue,"MIKE-Blue_Quartz_master.fits",clobber=args.clobber)

    # Red quartz
    master_quartz_red = median_combine("MIKE-Red_Quartz_list.txt","Quartz flat - Red",args.verbose)
    rough_tracks_red = track_orders(master_quartz_red,verbose=args.verbose)
    smoothed_tracks_red = smooth_tracks(master_quartz_red,rough_tracks_red)
    order_masks_red = generate_order_mask(smoothed_tracks_red,master_quartz_red)
    pickle.dump(order_masks_red,open("MIKE-Red_order_masks.pickle","wb"))
    save_fits(master_quartz_red,"MIKE-Red_Quartz_master.fits",clobber=args.clobber)

    # Blue milky
    master_milky_blue = median_combine("MIKE-Blue_Milky_list.txt","Milky flat - Blue",args.verbose)
    save_fits(master_milky_blue,"MIKE-Blue_Milky_master.fits",clobber=args.clobber)

    # Red milky
    master_milky_red = median_combine("MIKE-Red_Milky_list.txt","Milky flat - Red",args.verbose)
    save_fits(master_milky_red,"MIKE-Red_Milky_master.fits",clobber=args.clobber)

    # Blue arc
    master_arc_blue = median_combine("MIKE-Blue_ThAr_list.txt","Arc lamps - Blue",args.verbose)
    save_fits(master_arc_blue,"MIKE-Blue_ThAr_master.fits",clobber=args.clobber)

    # Red arc
    master_arc_red = median_combine("MIKE-Red_ThAr_list.txt","Arc lamps - Red",args.verbose)
    save_fits(master_arc_red,"MIKE-Red_ThAr_master.fits",clobber=args.clobber)

    return

if __name__ == "__main__":
    main()
