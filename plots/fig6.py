
import os
import numpy as np
import cv2

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors as mcolors

from method2color import method2color_d

SMALL_SIZE = 10
MEDIUM_SIZE = 20
BIG_SIZE = 25

plt.rc('font', size=10)
plt.rc('axes', titlesize=10)
plt.rc('legend', fontsize=9)


scale_l = [1.25, 1.5, 1.75, 2.]
method_l = ['sift', 'elf-vgg', 'superpoint', 'lfnet', 'lift']
trials_l = ['48', '12', '15', '4', '18']
color_l = [method2color_d[method] for method in method_l]
alpha=1

plt.figure(1, figsize=(8, 3))
G = gridspec.GridSpec(1,2)

ax1 = plt.subplot(G[0, 0])
ax1.set_ylabel('Repeatability')
ax1.set_xlabel('Scale factor')
ax1.set_title('HPatches scale: Repeatability')

ax2 = plt.subplot(G[0, 1])
ax2.set_ylabel('Matching score')
ax2.set_xlabel('Scale factor')
ax2.set_title('HPatches scale: Matching score')

for i, method in enumerate(method_l):
    trials  = trials_l[i]
    color   = color_l[i]
    if method=='elf-vgg':
        res_dir = os.path.join('res/elf', trials)
    else:
        res_dir = os.path.join('res', method, trials)
    #print('method: %s - metric: %s'%(method, metric))
    log_fn = os.path.join(res_dir, 'log.txt')
    log_l = [l.split("\n")[0] for l in open(log_fn).readlines() ]

    rep, ms = [],[]
    for l in log_l:
        if 'rep:' in l:
            #print(l)
            rep.append(float(l.split("-")[0].split(":")[1]))

        if 'ms:' in l:
            #print(l)
            ms.append(float(l.split("-")[0].split(":")[1]))

    rep = np.array(rep)
    rep = np.reshape(rep, (-1,4))
    
    mean_rep_per_scale = np.mean(rep, axis=0)
    ms = np.reshape(ms, (-1,4))
    mean_ms_per_scale = np.mean(ms, axis=0)

        
    ax1.plot(scale_l, mean_rep_per_scale, color, label=method, alpha=alpha)
    ax2.plot(scale_l, mean_ms_per_scale, color, label=method, alpha=alpha)
 
s_max = np.max(scale_l)
ax1.axis([1.2, s_max, 0, 1])
ax2.axis([1.2, s_max, 0, 1])

ax1.legend(loc=3)
plt.tight_layout()
plt.savefig('fig/fig6.png')
plt.close()

toto = cv2.imread('fig/fig6.png')
cv2.imshow('fig6', toto)
cv2.waitKey(0)


