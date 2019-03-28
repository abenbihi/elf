

import numpy as np
import cv2

import torch.nn.functional

from tools.cst import myjet


def SuperPoint_interpolate(pts, desc, W, H):
    """
    Copied/extracted from SuperPoint official code.
    Args:
        pts: shape [3,#kp]
        H,W : height, width of the image
    """
    desc = np.transpose(desc, (2,0,1))
    desc = np.expand_dims(desc, 0)
    D = desc.shape[1] # C of the feature map
    samp_pts = torch.from_numpy(pts[:2, :].copy()) # (2,#kp)
    coarse_desc = torch.from_numpy(desc.copy()) # (2,#kp)
    # bring to [-1,1] because that is what torch wants
    samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
    samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
    samp_pts = samp_pts.transpose(0, 1).contiguous()
    samp_pts = samp_pts.view(1, 1, -1, 2) #(1,1,#kp,2)
    samp_pts = samp_pts.float()
    desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
    desc = desc.data.cpu().numpy().reshape(D, -1)
    desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    desc = np.transpose(desc)
    return desc


def nms_fast(gradmap, conf_thresh, nms_dist, border_remove,
        max_num_features=1000):
    """
    Copy paste from SuperPoint
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      pts - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    H,W = gradmap.shape[:2]
    #print(np.where(gradmap>=conf_thresh))
    xs, ys = np.where(gradmap>conf_thresh) # TODO TODO TODO TODO TODO 
    if len(xs) == 0:
      return np.zeros((3, 0)), None, None
    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = gradmap[xs, ys]
    
    # begin nms_fast
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-pts[2,:])
    corners = pts[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, pts[2])).reshape(3,1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1,i], rcorners[0,i]] = 1
        inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = nms_dist
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0]+pad, rc[1]+pad)
        if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
            grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    #return out, out_inds

    # post nms_fast
    pts = out
    inds = np.argsort(pts[2,:])
    pts = pts[:,inds[::-1]] # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    #print(pts.shape)
    pts = pts[:,:max_num_features]
    return pts



def kapur_thresh(heatmap, blur=(1==1), k_size=5, sigma=0):
    """
    Shamelessly copied-pasted from 
    https://github.com/zenr/ippy/blob/master/segmentation/max_entropy.py
    Args:
        heatmap: float map ov values in [0,1]
    """
    """
    Implements Kapur-Sahoo-Wong (Maximum Entropy) thresholding method
    Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for Gray-Level Picture Thresholding Using the Entropy
    of the Histogram", Graphical Models and Image Processing, 29(3): 273-285
    M. Emre Celebi
    06.15.2007
    Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
    2016-04-28: Adapted for Python 2.7 by Robert Metchev from Java source of MaxEntropy() in the Autothresholder plugin
    http://rsb.info.nih.gov/ij/plugins/download/AutoThresholder.java
    :param data: Sequence representing the histogram of the image
    :return threshold: Resulting maximum entropy threshold
    """
    
    heatmap_uint8 = (255.0*heatmap).astype(np.uint8)
    if blur:
        #k_size = 15
        #sigma = 3
        heatmap_uint8 = cv2.GaussianBlur(heatmap_uint8,(k_size,k_size),sigma)

    data = np.histogram(heatmap_uint8, bins=256, range=(0, 256))[0]

    # calculate CDF (cumulative density function)
    cdf = data.astype(np.float).cumsum()

    # find histogram's nonzero area
    valid_idx = np.nonzero(data)[0]
    first_bin = valid_idx[0]
    last_bin = valid_idx[-1]

    # initialize search for maximum
    max_ent, threshold = 0, 0

    for it in range(first_bin, last_bin + 1):
        # Background (dark)
        hist_range = data[:it + 1]
        hist_range = hist_range[hist_range != 0] / cdf[it]  # normalize within selected range & remove all 0 elements
        tot_ent = -np.sum(hist_range * np.log(hist_range))  # background entropy

        # Foreground/Object (bright)
        hist_range = data[it + 1:]
        # normalize within selected range & remove all 0 elements
        hist_range = hist_range[hist_range != 0] / (cdf[last_bin] - cdf[it])
        tot_ent -= np.sum(hist_range * np.log(hist_range))  # accumulate object entropy

        # find max
        if tot_ent > max_ent:
            max_ent, threshold = tot_ent, it

    threshold /= 255.0
    return threshold


def preproc(img):
    """
    Preprocess img to feed to net: standardization i.e. (img-mean)/stddev.
    Returns img to feed to the net of shape
    [1,h,w,c]. 
    Args:
        img: img of shape [h,w,c]

    """
    if img.ndim == 3 and img.shape[-1] == 3:
        patch = img.astype(np.float32)
        mean = np.mean(patch, axis=(0,1,2))
        size = np.prod(np.array(patch.shape))
        stddev = np.maximum(np.std(patch, axis=(0,1,2)), 1.0/np.sqrt(size*size))
        patch = (patch-mean)/stddev
        patch   = np.expand_dims(patch, 0)
    else:
        patch = img.astype(np.float32)
        mean = np.mean(patch, axis=(0,1))
        size = np.prod(np.array(patch.shape))
        stddev = np.maximum(np.std(patch, axis=(0,1)), 1.0/np.sqrt(size*size))
        patch = (patch-mean)/stddev
        patch   = np.expand_dims(patch, 0)
        patch   = np.expand_dims(patch, 3)

    return patch

def postproc(grad, noise_k_size, noise_sigma, thr_k_size, thr_sigma,
        nms_dist, border_remove, max_num_feat):
    """
    Detect keypoints from grad
    """
    

    grad =  np.abs(grad) # keep high intensity grads whether >0 or <0
    if grad.ndim == 3 and grad.shape[-1] == 3: # color grad
        grad = np.mean(grad, axis=2) # average across channels
    grad /= np.max(grad) # normalize

    # denoising
    grad_denoised = (255*grad).astype(np.uint8)
    k = noise_k_size
    blur_k = (k,k)
    grad_denoised = cv2.GaussianBlur(
            grad_denoised, blur_k, noise_sigma)
    grad_denoised = grad_denoised/255.0

    # thresholding
    do_blur = (1==1)
    min_conf = kapur_thresh(grad, do_blur, thr_k_size, thr_sigma)
    grad_threshed = grad_denoised.copy()
    grad_threshed[grad_threshed<min_conf] = min_conf
    
    # heatmap visualisation
    heatmap = -np.log(grad_threshed + 0.0001)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
    heatmap = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
    heatmap = (heatmap*255).astype('uint8')

    # nms: ignore maxima below min_conf
    pts = nms_fast(grad_threshed, min_conf, nms_dist, border_remove, 
            max_num_feat)

    # check if kp were detected
    if isinstance(pts, tuple):
        pts_fail = True
    elif (pts.shape==(3,) or pts.shape==(3,1) 
            or pts.shape==(2,) or pts.shape==(3,0)): # empty nms
        pts_fail = True
    else:
        pts_fail = False
    

    return pts_fail, pts

