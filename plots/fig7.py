
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

angle_v = [a for a in range(10,211,40)]
method_l = ['sift', 'elf-vgg', 'superpoint', 'lfnet', 'lift']
trials_l = ['47', '11', '14', '6', '19']
color_l = [method2color_d[method] for method in method_l]
alpha=1

plt.figure(1, figsize=(8, 3))
G = gridspec.GridSpec(1, 2)

ax1 = plt.subplot(G[0, 0])
ax1.set_ylabel('Repeatability')
ax1.set_xlabel('Angle (°)')
ax1.set_title('HPatches rotation: Repeatability')

ax2 = plt.subplot(G[0, 1])
ax2.set_ylabel('Matching score')
ax2.set_xlabel('Angle (°)')
ax2.set_title('HPatches rotation: Matching score')

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
    rep = np.reshape(rep, (-1, len(angle_v)))
    mean_rep_per_angle = np.mean(rep, axis=0)
    ms = np.reshape(ms, (-1,len(angle_v)))
    mean_ms_per_angle = np.mean(ms, axis=0)

        
    ax1.plot(angle_v, mean_rep_per_angle, color, label=method, alpha=alpha)
    ax2.plot(angle_v, mean_ms_per_angle, color, label=method, alpha=alpha)

a_max = np.max(angle_v)
ax1.axis([0, a_max, 0, 1])
ax2.axis([0, a_max, 0, 1])
ax1.legend(loc=3)
plt.tight_layout()
plt.savefig('fig/fig7_angle.png')
plt.close()

toto = cv2.imread('fig/fig7_angle.png')
cv2.imshow('toto_angle', toto)
cv2.waitKey(0)

