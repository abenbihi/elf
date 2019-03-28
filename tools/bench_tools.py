
import time
import numpy as np
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics

def kp_map(shape, kp_l):
    """
    Draws kp locations on black-white map.
    Args:
        shape: (j,i):(x,y) convention
        kp_l: (j,i):(x.y) convention
    """
    map_ = np.zeros((shape[1], shape[0]))
    for kp in kp_l:
        x,y = kp.pt
        #print(kp.pt)
        map_[int(y),int(x)] = 255
    map_ = map_.astype(np.uint8)
    return map_

def cv_hamming_cost_matrix(des1, des2):
    """
    Computes cost matrix of hamming distance between every descriptor.
    Bad bad bad me :( I am doing some ugly loop stuff 
    I am doing this because by hamming norm and the one from opencv look
    different.
    """
    desc_norm = np.zeros((des1.shape[0], des2.shape[0]))
    for i in range(des1.shape[0]):
        for j in range(des2.shape[0]):
            desc_norm[i,j] = cv2.norm(des1[i,:], des2[j,:], cv2.NORM_HAMMING)
    return desc_norm

def greedy_bipartite_matching(cost_matrix, thresh): #, N1, N2):
    """
    Runs greedy bipartitie matrching on the cost matrix 'norm'. 
    Returns the list of correspondence idx in
    [0,norm.shape[0]]x[0,norm.shape[1]].
    Args:
        norm: cost matrix
        thresh: threshold value above which a candidate match is discarded
        (N1: number of features in img1 (row of the cost matrix))
        (N2: number of features in img2 (col of the cost matrix))
    """
    N1 = cost_matrix.shape[0]
    N2 = cost_matrix.shape[1]
    # Candidate matches are kp which distance (in pixel or in descriptor space)
    # is below the threshold.
    cand_match_i, cand_match_j = np.where(cost_matrix<=thresh)
    cand_cost = cost_matrix[cand_match_i, cand_match_j]
    argsort = np.argsort(cand_cost)
    cand_match_i = cand_match_i[argsort]
    cand_match_j = cand_match_j[argsort]
    cand_cost = cand_cost[argsort]
    
    # Each kp must have at most one match. To do so, associate kp in increasing
    # distance order until they are all matched.
    is_matched_i = np.zeros(N1)
    is_matched_j = np.zeros(N2)
    max_num_match = np.minimum(N1, N2)
    corr = 0 # number of correspondences
    corr_idx = [] # list of correspondences
    for idx in range(cand_cost.shape[0]):
        i = cand_match_i[idx]
        j = cand_match_j[idx]
        if is_matched_i[i]==1 or is_matched_j[j]==1:
            continue
        # neither i nor j are matched yet
        is_matched_i[i]=1 
        is_matched_j[j]=1
        corr_idx.append((i,j))
        corr+=1

        if corr >= max_num_match:
            break

    return corr_idx

def rep(shape, H, kp1, kp2, thresh_overlap):
    """ OK
    Compute the repeatability. Implementation of greedy bipartite mattching.
    Slightly faster than my_greedy_rep. 
    Args:
        shape: img shape (x,y):(j,i)
        H: ground truth H (such that img2 = H(img1))
        kp1: list of cv.keypoints in img1
        kp2: list of cv.keypoints in img2
        thresh_overlap: euclidean distance above which 2 keypoints do not match
    """
    H_inv = np.linalg.inv(H)
    h_pts1 = np.vstack([[kp.pt[0],kp.pt[1], 1] for kp in kp1])
    h_pts2 = np.vstack([[kp.pt[0],kp.pt[1], 1] for kp in kp2])
    #print(h_pts2.shape)
    
    # Filter out kp2 such that H^-1*kp2 is outside shape
    h_w_1_pts2 = np.transpose(np.matmul(H_inv, np.transpose(h_pts2)))
    w_1_pts2 = (h_w_1_pts2/np.expand_dims(h_w_1_pts2[:,2],1))[:,[0,1]]

    mask2 = (w_1_pts2[:, 0]>=0) & (w_1_pts2[:, 0]<shape[0]) &\
            (w_1_pts2[:, 1]>=0) & (w_1_pts2[:, 1]<shape[1])
    pts2 = h_pts2[mask2,:][:,[0,1]]
    
    # Filter out kp1 such that H*kp1 is outside shape
    h_w_pts1 = np.transpose(np.matmul(H, np.transpose(h_pts1)))
    w_pts1 = (h_w_pts1/np.expand_dims(h_w_pts1[:,2],1))[:,[0,1]]
    mask1 = (w_pts1[:, 0]>=0) & (w_pts1[:, 0]<shape[0]) &\
            (w_pts1[:, 1]>=0) & (w_pts1[:, 1]<shape[1])
    w_pts1 = w_pts1[mask1,:]

    # run greedy bipartite matching between the kp   
    N1 = w_pts1.shape[0]
    N2 = pts2.shape[0]
    denom = np.minimum(N1, N2)
    if denom==0:
        rep = 0
        M, M_d, inter = [],[],[]
    else:
        w_pts1 = np.expand_dims(w_pts1, 1) # row
        pts2 = np.expand_dims(pts2, 0) # col
        kp_norm = np.linalg.norm(w_pts1 - pts2, ord=None, axis=2)

        start_time = time.time()
        M = greedy_bipartite_matching(kp_norm, thresh_overlap)# correspondences
        rep = 1.0*len(M)/np.minimum(N1, N2) # repeatability 
         
        #duration = time.time() - start_time
        #minutes = duration / 60
        #seconds = duration % 60
        #print('rep: %d min %d s - cor_num:%d - feat_num: %d - rep:%.10f' 
        #        %(minutes, seconds, M, denom, rep))

    return rep,N1,N2,M #len(M)

def ms(shape, H, kp1, kp2, des1, des2, thresh_overlap, thresh_desc, norm='L2'):
    """
    Matching score. Run greedy bipartite matching both on kp and desc.
    Args:
        shape: img shape (x,y):(j,i)
        H: ground truth H (such that img2 = H(img1))
        kp1: list of cv.keypoints in img1
        kp2: list of cv.keypoints in img2
        des1: list of descriptor for kp in kp1
        des2: list of descriptor for kp in kp2
        thresh_overlap: euclidean distance above which 2 keypoints do not match
        thresh_desc: descriptor L2 distance above which 2 keypoints do not match
    """
    if len(kp1)==0 or len(kp2)==0:
        N1 = len(kp1)
        N2 = len(kp2)
        ms = 0
        M, M_d, inter = [],[],[]
        return ms, N1, N2, len(M), len(M_d), len(inter)

    H_inv = np.linalg.inv(H)
    h_pts1 = np.vstack([[kp.pt[0],kp.pt[1], 1] for kp in kp1])
    h_pts2 = np.vstack([[kp.pt[0],kp.pt[1], 1] for kp in kp2])
    #des1 = np.vstack(des1)
    #des2 = np.vstack(des2)
    #print('h_pts2.shape: ', h_pts2.shape)
    #print('des2.shape:', des2.shape)
    #print('h_pts1.shape: ', h_pts1.shape)
    #print('des1.shape:', des1.shape)

    # Filter out kp2 such that H^-1*kp2 is outside shape
    h_w_1_pts2 = np.transpose(np.matmul(H_inv, np.transpose(h_pts2)))
    w_1_pts2 = (h_w_1_pts2/np.expand_dims(h_w_1_pts2[:,2],1))[:,[0,1]]
    
    mask2 = (w_1_pts2[:, 0]>=0) & (w_1_pts2[:, 0]<shape[0]) &\
            (w_1_pts2[:, 1]>=0) & (w_1_pts2[:, 1]<shape[1])
    pts2 = h_pts2[mask2,:][:,[0,1]]
    des2 = des2[mask2,:]
    
    # Filter out kp1 such that H*kp1 is outside shape
    h_w_pts1 = np.transpose(np.matmul(H, np.transpose(h_pts1)))
    w_pts1 = (h_w_pts1/np.expand_dims(h_w_pts1[:,2],1))[:,[0,1]]
    mask1 = (w_pts1[:, 0]>=0) & (w_pts1[:, 0]<shape[0]) &\
            (w_pts1[:, 1]>=0) & (w_pts1[:, 1]<shape[1])
    w_pts1 = w_pts1[mask1,:]
    des1 = des1[mask1,:]

    # run greedy bipartite matching between the kp   
    N1 = w_pts1.shape[0]
    N2 = pts2.shape[0]
    denom = np.minimum(N1, N2)
    if denom==0:
        ms = 0
        M, M_d, inter = [],[],[]
    else:
        w_pts1 = np.expand_dims(w_pts1, 1) # row
        pts2 = np.expand_dims(pts2, 0) # col
        kp_norm = np.linalg.norm(w_pts1 - pts2, ord=None, axis=2)

        if norm=='L2':
            des1 = np.expand_dims(des1, 1) # row
            des2 = np.expand_dims(des2, 0) # col
            desc_norm = np.linalg.norm(des1 - des2, ord=None, axis=2)
        elif norm=='hamming':
            desc_norm = cv_hamming_cost_matrix(des1, des2)
        else:
            print('Error: unknown norm type: %s'%norm)
            exit(1)

        start_time = time.time()
        M = greedy_bipartite_matching(kp_norm, thresh_overlap)
        M_d = greedy_bipartite_matching(desc_norm, thresh_desc)
        inter = [l for l in M if l in M_d]
        ms = 1.0*len(inter)/denom
         
        #duration = time.time() - start_time
        #minutes = duration / 60
        #seconds = duration % 60
        #print('rep: %d min %d s - cor_num:%d - feat_num: %d - rep:%.10f' 
        #        %(minutes, seconds, M, denom, rep))

    return ms, N1, N2, len(M), len(M_d), len(inter)


def mAP_NN(shape, H, kp1, kp2, des1, des2, thresh_overlap, thresh_desc_v, norm='L2'):
    """
    Compute the mAP on all detected feature, i.e. wihtout the matching
    filtering. 
    Args:
        shape: img shape (x,y):(j,i)
        H: ground truth H (such that img2 = H(img1))
        kp1: list of cv.keypoints in img1
        kp2: list of cv.keypoints in img2
        des1: list of descriptor for kp in kp1
        des2: list of descriptor for kp in kp2
        thresh_overlap: euclidean distance above which 2 keypoints do not match
        thresh_desc: descriptor L2 distance above which 2 keypoints do not match
        norm: L2 or hamming
    """
    if len(kp1)==0 or len(kp2)==0:
        N1 = len(kp1)
        N2 = len(kp2)
        ms = 0
        M, M_d, inter = [],[],[]
        return ms, N1, N2, len(M), len(M_d), len(inter)

    H_inv = np.linalg.inv(H)
    h_pts1 = np.vstack([[kp.pt[0],kp.pt[1], 1] for kp in kp1])
    h_pts2 = np.vstack([[kp.pt[0],kp.pt[1], 1] for kp in kp2])
    des1 = np.vstack(des1)
    des2 = np.vstack(des2)
    
    # Filter out kp2 such that H^-1*kp2 is outside shape
    h_w_1_pts2 = np.transpose(np.matmul(H_inv, np.transpose(h_pts2)))
    w_1_pts2 = (h_w_1_pts2/np.expand_dims(h_w_1_pts2[:,2],1))[:,[0,1]]

    mask2 = (w_1_pts2[:, 0]>=0) & (w_1_pts2[:, 0]<shape[0]) &\
            (w_1_pts2[:, 1]>=0) & (w_1_pts2[:, 1]<shape[1])
    pts2 = h_pts2[mask2,:][:,[0,1]]
    des2 = des2[mask2,:]
    
    # Filter out kp1 such that H*kp1 is outside shape
    h_w_pts1 = np.transpose(np.matmul(H, np.transpose(h_pts1)))
    w_pts1 = (h_w_pts1/np.expand_dims(h_w_pts1[:,2],1))[:,[0,1]]
    mask1 = (w_pts1[:, 0]>=0) & (w_pts1[:, 0]<shape[0]) &\
            (w_pts1[:, 1]>=0) & (w_pts1[:, 1]<shape[1])
    w_pts1 = w_pts1[mask1,:]
    des1 = des1[mask1,:]

    w_pts1 = np.expand_dims(w_pts1, 1) # row
    pts2 = np.expand_dims(pts2, 0) # col
    kp_norm = np.linalg.norm(w_pts1 - pts2, ord=None, axis=2)

    if norm=='L2':
        des1 = np.expand_dims(des1, 1) # row
        des2 = np.expand_dims(des2, 0) # col
        desc_norm = np.linalg.norm(des1 - des2, ord=None, axis=2)
    elif norm=='hamming':
        desc_norm = cv_hamming_cost_matrix(des1, des2)
    else:
        print('Error: unknown norm type: %s'%norm)
        exit(1)
    
    match = np.argmin(desc_norm, axis=1)
    P = np.sum( np.min(kp_norm, axis=1)<thresh_overlap)
    
    recall_v, precision_v = [],[]
    for thresh_desc in thresh_desc_v:
        recall, precision = 0,0
        # This is wrong, this gives a square matrix and not a vector
        #print((desc_norm[:, match]<thresh_desc).shape) 

        TP = np.sum((desc_norm[np.arange(desc_norm.shape[0]), match]<thresh_desc)*
                (kp_norm[np.arange(desc_norm.shape[0]),match]<thresh_overlap))
        FP = np.sum((desc_norm[np.arange(desc_norm.shape[0]), match]<thresh_desc)*
                ( 1-(kp_norm[np.arange(desc_norm.shape[0]),match]<thresh_overlap) ))
        #print('thresh_desc:%.2f - P:%d - TP:%d - FP:%d' 
        #        %(thresh_desc, P, TP, FP))
        #raw_input('wait')

        if P!=0:
            recall = 1.0*TP/P
        if (TP+FP)!=0:
            precision = 1.0*TP/(TP+FP)

        recall_v.append(recall)
        precision_v.append(precision)
    recall_v = np.array(recall_v)
    precision_v = np.array(precision_v)
    mAP = metrics.auc(recall_v, precision_v)
    #mAP =  np.sum(precision_v[1:] * (recall_v[1:] - recall_v[:-1]))

    return mAP, precision_v, recall_v


def mAP_NNDR(shape, H, kp1, kp2, des1, des2, thresh_overlap, thresh_desc_v):
    """
    Correspondence: 2 kp are a correpondence if their distance is below
    thresh_overlap.
    Matches: The set of regions A and B is img1 and img2 after NNDR matching on
    the descriptor space.
    Correct match: if 2 regions matching are a correspondence
    False match: if 2 regions matching are not a correspondence
    """

    if len(kp1)==0 or len(kp2)==0:
        N1 = len(kp1)
        N2 = len(kp2)
        ms = 0
        M, M_d, inter = [],[],[]
        return mAP 

    H_inv = np.linalg.inv(H)
    h_pts1 = np.vstack([[kp.pt[0],kp.pt[1], 1] for kp in kp1])
    h_pts2 = np.vstack([[kp.pt[0],kp.pt[1], 1] for kp in kp2])
    des1 = np.vstack(des1)
    des2 = np.vstack(des2)
    
    # Filter out kp2 such that H^-1*kp2 is outside shape
    h_w_1_pts2 = np.transpose(np.matmul(H_inv, np.transpose(h_pts2)))
    w_1_pts2 = (h_w_1_pts2/np.expand_dims(h_w_1_pts2[:,2],1))[:,[0,1]]
    mask2 = (w_1_pts2[:, 0]>=0) & (w_1_pts2[:, 0]<shape[0]) &\
            (w_1_pts2[:, 1]>=0) & (w_1_pts2[:, 1]<shape[1])
    pts2 = h_pts2[mask2,:][:,[0,1]]
    des2 = des2[mask2,:]
    
    # Filter out kp1 such that H*kp1 is outside shape
    h_w_pts1 = np.transpose(np.matmul(H, np.transpose(h_pts1)))
    w_pts1 = (h_w_pts1/np.expand_dims(h_w_pts1[:,2],1))[:,[0,1]]
    mask1 = (w_pts1[:, 0]>=0) & (w_pts1[:, 0]<shape[0]) &\
            (w_pts1[:, 1]>=0) & (w_pts1[:, 1]<shape[1])
    w_pts1 = w_pts1[mask1,:]
    des1 = des1[mask1,:]
   
    # if there is not feature (too bad)
    N1 = w_pts1.shape[0]
    N2 = pts2.shape[0]
    denom = np.minimum(N1, N2)
    mAP = 0
    if denom!=0:
        # correspondence
        w_pts1 = np.expand_dims(w_pts1, 1) # row
        pts2 = np.expand_dims(pts2, 0) # col
        kp_norm = np.linalg.norm(w_pts1 - pts2, ord=None, axis=2)
        P = np.sum(kp_norm<thresh_overlap)

        # NNDR matching
        good = [] # matching features according to desc. distance
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1, des2,k=2)
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance: # Lowe's ratio
                good.append(m)
        kp1_good    = [ kp1[m.queryIdx] for m in good ]
        des1_good   = np.vstack([ des1[m.queryIdx] for m in good ])
        kp2_good    = [ kp2[m.trainIdx] for m in good ]
        des2_good   = np.vstack([ des2[m.trainIdx] for m in good ])
        desc_norm_good = np.array([m.distance for m in good] )
        kp_norm_good = np.array([ kp_norm[m.queryIdx, m.trainIdx] for m in good])
        
        # recall, precision
        recall_v, precision_v = [],[]
        for thresh_desc in thresh_desc_v:
            recall, precision = 0,0
            TP = np.sum((desc_norm_good<thresh_desc)*(kp_norm_good<thresh_overlap))
            FP = np.sum((desc_norm_good<thresh_desc)*((1-kp_norm_good<thresh_overlap)))

            if P!=0:
                recall = 1.0*TP/P
            if (TP+FP)!=0:
                precision = 1.0*TP/(TP+FP)

            recall_v.append(recall)
            precision_v.append(precision)
    recall_v = np.array(recall_v)
    precision_v = np.array(precision_v)
    mAP =  np.sum(precision_v[1:] * (recall_v[1:] - recall_v[:-1]))

    return mAP, precision_v, recall_v
























