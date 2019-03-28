
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


scene_l = ['Fountain', 'Entry', 'Hersjesu']
img_num_l = [10, 9, 7]
method_l = ['sift', 'elf-vgg', 'superpoint', 'lfnet', 'lift']
trials_l = ['45', '15', '13', '8', '20']
color_l = [method2color_d[method] for method in method_l]
alpha=1

plt.figure(1, figsize=(15, 6))
G = gridspec.GridSpec(2,3)

ax_d = {}
for i in range(2): # line0:rep, line1: ms
    ax_d[i] = {}
    for j in range(3):
        ax_d[i][j] = plt.subplot(G[i,j])
        if i==0:
            ax_d[i][j].set_ylabel('Repeatability')
        else:
            ax_d[i][j].set_ylabel('Matching score')
        ax_d[i][j].set_title('%s'%scene_l[j])
        ax_d[i][j].axis([0, img_num_l[j], 0, 1])

for m, method in enumerate(method_l):
    trials  = trials_l[m]
    color   = color_l[m]
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
    
    count_b, count_e = 0,0
    for j,scene in enumerate(scene_l):
        count_e += img_num_l[j]
        #print(count_b, count_e)
        absc = range(img_num_l[j])
        #print(absc)
        #print(rep[count_b:count_e])
        ax_d[0][j].plot(absc, rep[count_b:count_e], color, label=method, alpha=alpha)
        ax_d[1][j].plot(absc, ms[count_b:count_e], color, label=method, alpha=alpha)
        count_b = count_e

for i in range(2): # line0:rep, line1: ms
    for j in range(3):
        ax_d[i][j].legend(loc=0)

plt.tight_layout()
plt.savefig('fig/fig8.png')
plt.close()

toto = cv2.imread('fig/fig8.png')
cv2.imshow('toto_strecha', toto)
cv2.waitKey(0)


