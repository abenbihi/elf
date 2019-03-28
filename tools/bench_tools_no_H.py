
import time
import numpy as np
import cv2

from tools.bench_tools import greedy_bipartite_matching, cv_hamming_cost_matrix

##########################################################
# for img pairs with transformation!=homography

# copied from colmap
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


# copied from colmap
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

# copied from colmap
def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def load_camera(fn):
    data = [l.split("\n")[0].split(" ")[:-1] for l in open(fn).readlines() ]
    K = data[:3]
    K = np.zeros((3,3))
    K[0:3,0:3] = np.reshape(data[:3], (3,3))
    
    R = np.reshape(data[4:7], (3,3)).astype(np.float32)
    R = R.T # because they store R transpose, who the fuck does that ?
    c = np.reshape(data[7], (3)).astype(np.float32) # this file is fucked up ...
    t = np.matmul(-R,c)

    T = np.zeros((4,4))
    T[0:3,0:3] = R
    T[0:3,3] = t*1e-2
    T[3,3] = 1
    return K,T

def project(h_pts0, K, depth0_map, R0, t0, R1, t1):
    """
    Project keypoints from img0 to img1.
    Return warped keypoints in the form (2,-1)
    Args:
        kp0: kp list (opencv like i.e. kp.pt is (x,y)
        h_pts0: array homogenous pixels (x,y,1) with shape (3,-1)
        K: intrinsic camera (I assume it is the same for both img)
        depth0_map: depth map of img0
        R0: camera0 rotation from the camera to the world
        t0: camera0 translation from the camera to the world
        R1: camera1 rotation from the camera to the world
        t1: camera1 translation from the camera to the world
    """
    K_inv  = np.linalg.inv(K)
    #h_pts0 = np.vstack([[kp.pt[0],kp.pt[1], 1] for kp in kp0]).T
    
    # image0 [u,v,1] -> camera frame 0 [x/z,y/z,1] 
    pts0_c0 = np.dot(K_inv, h_pts0)
    
    # specify scale -> camera frame 0[x,y,z] 
    l0, c0 = h_pts0[1,:].astype(np.int), h_pts0[0,:].astype(np.int)
    depth0 = depth0_map[l0,c0]
    pts0_c0 = pts0_c0*depth0
    
    # camera frame 0 [x,y,z] -> world frame [x,y,z] #
    T = np.zeros((4,4))
    T[0:3,0:3] = R0 # Assumes R = R c->w
    T[0:3,3] = t0 # Assumes t = t c->w
    T[3,3] = 1
    pts0_c0 = np.vstack((pts0_c0, np.ones(pts0_c0.shape[1])))
    pts0_w = np.dot(np.linalg.inv(T), pts0_c0)
    
    # world frame [x,y,z] -> camera frame 2 [x,y,z]
    T = np.zeros((4,4))
    T[0:3,0:3] = R1 # R from frame 0 to frame n 
    T[0:3,3] = t1 # t from frame 0 to frame n
    T[3,3] = 1
    pts0_c1 = np.dot(T, pts0_w) # shape (4,-1)
    
    # camera frame n -> image_n [u,v,1]
    h_w_pts0 = np.dot(K, pts0_c1[:3,:]) # shape (3,-1), (x,y)
    w_pts0 = (h_w_pts0/np.expand_dims(h_w_pts0[2,:],0))[[0,1],:] # (y,x)

    return w_pts0


def rep(shape, h_pts0, h_pts1, K, depth0_map, depth1_map, R0, t0, R1, t1,
        thresh_overlap):
    # Filter out kp0 such that warped kp0 is outside shape
    #w_pts0 = project(kp0, K, depth0_map, R0, t0, R1, t1)
    w_pts0 = project(h_pts0.T, K, depth0_map, R0, t0, R1, t1)
    #print('pre-mask: w_pts0.shape', w_pts0.shape)
    mask0 = (w_pts0[0, :]>=0) & (w_pts0[0, :]<shape[0]) &\
            (w_pts0[1, :]>=0) & (w_pts0[1, :]<shape[1])
    #print(mask0.shape)
    w_pts0 = w_pts0[:,mask0].T # (-1,2)

    # Filter out kp0 such that warped kp0 is outside shape
    #w_pts1 = project(kp1, K, depth1_map, R1, t1, R0, t0)
    w_pts1 = project(h_pts1.T, K, depth1_map, R1, t1, R0, t0)
    #print('pre-mask: w_pts1.shape', w_pts1.shape)
    mask1 = (w_pts1[0, :]>=0) & (w_pts1[0, :]<shape[0]) &\
            (w_pts1[1, :]>=0) & (w_pts1[1, :]<shape[1])
    #print(mask1.shape)
    w_pts1 = w_pts1[:,mask1].T # (-1,2)
    pts1 = h_pts1[mask1,:][:,[0,1]]

    # run greedy bipartite matching between the kp   
    N1 = w_pts0.shape[0]
    N2 = pts1.shape[0]
    denom = np.minimum(N1, N2)
    if denom==0:
        rep = 0
        M, M_d, inter = [],[],[]
    else:
        w_pts0 = np.expand_dims(w_pts0, 1) # row
        pts1 = np.expand_dims(pts1, 0) # col
        kp_norm = np.linalg.norm(w_pts0 - pts1, ord=None, axis=2)

        start_time = time.time()
        M = greedy_bipartite_matching(kp_norm, thresh_overlap)# correspondences
        rep = 1.0*len(M)/np.minimum(N1, N2) # repeatability 

    #print('thresh: %.3f - rep: %.3f'%(thresh_overlap, rep))
    return rep,N1,N2,len(M)
 
def ms(shape, des0, des1, h_pts0, h_pts1, K, depth0_map, depth1_map, R0,
        t0, R1, t1, thresh_overlap, thresh_desc, norm='L2'):

    """
    Matching score. Run greedy bipartite matching both on kp and desc.
    Args:
        shape: img shape (x,y):(j,i)
        H: ground truth H (such that img2 = H(img1))
        kp0: list of cv.keypoints in img1
        kp1: list of cv.keypoints in img2
        des0: list of descriptor for kp in kp0
        des1: list of descriptor for kp in kp1
        thresh_overlap: euclidean distance above which 2 keypoints do not match
        thresh_desc: descriptor L2 distance above which 2 keypoints do not match
    """
    # Filter out kp0 such that warped kp0 is outside shape
    w_pts0 = project(h_pts0.T, K, depth0_map, R0, t0, R1, t1)
    #print('pre-mask: w_pts0.shape', w_pts0.shape)
    mask0 = (w_pts0[0, :]>=0) & (w_pts0[0, :]<shape[0]) &\
            (w_pts0[1, :]>=0) & (w_pts0[1, :]<shape[1])
    #print(mask0.shape)
    w_pts0 = w_pts0[:,mask0].T # (-1,2)
    des0 = des0[mask0,:]

    # Filter out kp0 such that warped kp0 is outside shape
    w_pts1 = project(h_pts1.T, K, depth1_map, R1, t1, R0, t0)
    #print('pre-mask: w_pts1.shape', w_pts1.shape)
    mask1 = (w_pts1[0, :]>=0) & (w_pts1[0, :]<shape[0]) &\
            (w_pts1[1, :]>=0) & (w_pts1[1, :]<shape[1])
    #print(mask1.shape)
    w_pts1 = w_pts1[:,mask1].T # (-1,2)
    pts1 = h_pts1[mask1,:][:,[0,1]]
    des1 = des1[mask1,:]

    # run greedy bipartite matching between the kp   
    N1 = w_pts0.shape[0]
    N2 = pts1.shape[0]
    denom = np.minimum(N1, N2)
    if denom==0:
        ms = 0
        M, M_d, inter = [],[],[]
    else:
        w_pts0 = np.expand_dims(w_pts0, 1) # row
        pts1 = np.expand_dims(pts1, 0) # col
        kp_norm = np.linalg.norm(w_pts0 - pts1, ord=None, axis=2)

        if norm=='L2':
            des0 = np.expand_dims(des0, 1) # row
            des1 = np.expand_dims(des1, 0) # col
            desc_norm = np.linalg.norm(des0 - des1, ord=None, axis=2)
        elif norm=='hamming':
            desc_norm = cv_hamming_cost_matrix(des0, des1)
        else:
            print('Error: unknown norm type: %s'%norm)
            exit(1)

        start_time = time.time()
        M = greedy_bipartite_matching(kp_norm, thresh_overlap)
        M_d = greedy_bipartite_matching(desc_norm, thresh_desc)
        inter = [l for l in M if l in M_d]
        ms = 1.0*len(inter)/denom
         
    return ms, N1, N2, len(M), len(M_d), len(inter)

