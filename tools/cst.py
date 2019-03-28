import numpy as np

DATA    = 'hpatches'
DEBUG   = (0==1)


WS_DIR = '/home/ws/'
WEIGHT_DIR = '%s/meta/weights/'%WS_DIR

NEW_SIZE = (640,480)
GLOBAL_TIME = True
MIN_MATCH_COUNT = 10 # min num of descriptor matching for H estimation
BATCH_SIZE = 1
MOVING_AVERAGE_DECAY = 0.9999

SCALE_NUM = 5
THRESH_OVERLAP = 5
THRESH_DESC = 10000


if DATA=='hpatches':
    DATA_DIR = '%s/datasets/hpatches-sequences-release/'%WS_DIR
    HP_LIST = 'meta/list/img_hp.txt'
    MAX_IMG_NUM = 6 # number of img per scene to process
    IMG_EXT = 'ppm'
    SCENE_LIST = [l.split("\n")[0] for l in open(HP_LIST).readlines() ]
elif DATA=='hpatches_rot':
    DATA_DIR = '%s/datasets/hpatches_rot/'%WS_DIR
    HP_LIST = 'meta/list/img_hp.txt'
    MAX_IMG_NUM = 7 # number of img per scene to process
    IMG_EXT = 'ppm'
elif DATA=='hpatches_s':
    DATA_DIR = '%s/datasets/hpatches_s/'%WS_DIR
    HP_LIST = 'meta/list/img_hp.txt'
    MAX_IMG_NUM = 5 # number of img per scene to process
    IMG_EXT = 'ppm'
elif DATA=='strecha':
    DATA_DIR = '%s/datasets/strecha/'%WS_DIR
    SCENE_LIST = ['fountain', 'castle_entry','herzjesu'] 
elif DATA=='webcam':
    SCENE_LIST = ['Chamonix', 'Courbevoie', 'Frankfurt', 'Mexico', 'Panorama', 'StLouis']
    DATA_DIR = '%s/datasets/WebcamRelease'%WS_DIR
else:
    print('Error: unknown dataset: %s. Set DATA correctly in tools/cst.py.'%DATA)
    exit(1)


# copied from superpoint
# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])
# lift
KP_DIR = 'kp'
ORI_DIR = 'ori'
DES_DIR = 'des'


# lfnet
LFNET_DIR = '/home/ws/methods/lfnet/lf-net-release/' # docker path
