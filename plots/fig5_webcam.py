
import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from method2color import method2color_d

SMALL_SIZE = 10
MEDIUM_SIZE = 20
BIG_SIZE = 25

plt.rc('font', size=10)
plt.rc('axes', titlesize=10)
plt.rc('legend', fontsize=BIG_SIZE)

width = 0.8
x_shift = 0.45

perf_d = {}
method_l = ['vgg_sal', 'alex_sal', 'xcep_sal', 'lfnet_sal', 'sp_sal', 'lfnet', 
        'superpoint', 'lift', 'sift', 'surf', 'orb', 'kaze', 'tilde', 'mser']
method_num = len(method_l)

my_method_num = 5 # there are 5 variants of my detector
absc1 = np.arange(my_method_num)
absc2 = np.arange(method_num-my_method_num) + np.max(absc1) + 3 
absc = np.hstack((absc1, absc2))

color_l = [method2color_d[method] for method in method_l]


# 1 chiffre apres la virgule
rep_l = [53.2, 38.5, 49.8, 41.9, 46.3, 48.3,
        46.4, 42.2,  33.9, 28.3, 31.6, 41.0, 52.5, 52.2]
ms_l = [43.7, 31.9, 35.5, 33.4, 18.1, 18.1, 
        32.4, 17.8, 10.1, 8.3, 1.3, 13.9, 34.7, 6.1]


plt.figure(1, figsize=(10, 3))
G = gridspec.GridSpec(1, 2)

ax1 = plt.subplot(G[0, 0])
ax1.set_ylabel('Repeatability')
ax1.set_xlabel('Methods')
ax1.set_title('Webcam: Repeatability')
ax1.bar(absc, rep_l, width, color=color_l)
ax1.set_xticks(absc)
ax1.set_xticklabels([], fontsize=6)
ax1.axis([-1, np.max(absc)+1, 0, 100])
for i, rep in enumerate(rep_l):
    ax1.text(absc[i]-x_shift, rep + 1, str(rep), color='black', fontsize=8)#, fontweight='bold')

net_fontsize = 15
ax1.text(-0.5, 85, 'Our variants', color='black', fontsize=net_fontsize, fontweight='bold')
ax1.text(9.5, 85, 'SOTA', color='black', fontsize=net_fontsize, fontweight='bold')


ax2 = plt.subplot(G[0, 1])
ax2.set_ylabel('Matching scores')
ax2.set_xlabel('Methods')
ax2.set_title('Webcam: Matching Score')
ax2.bar(absc, ms_l, width, color=color_l)
ax2.set_xticks(absc)
ax2.set_xticklabels([], fontsize=6)
ax2.axis([-1, np.max(absc)+1, 0, 100])

for i, ms in enumerate(ms_l):
    ax2.text(absc[i]-x_shift, ms + 1, str(ms), color='black', fontsize=8)

plt.tight_layout()
plt.savefig('fig/fig5_webcam.png')
plt.close()

toto = cv2.imread('fig/fig5_webcam.png')
cv2.imshow('fig9', toto)
cv2.waitKey(0)

