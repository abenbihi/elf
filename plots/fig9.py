
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

width = 1

x_shift_right = 0.
x_shift_left = 1.2
x_shift_right_2 = 1.05
bar_fontsize = 7

method_l = ['lfnet', 'superpoint', 'lift', 'sift', 'surf', 'orb']
method_num = len(method_l)
absc = np.linspace(0, method_num*2.8, method_num) + 3

color_l = [method2color_d[method] for method in method_l]
ecolor_l = ['black' for d in range(1, method_num+1)]

##############################################################################
# hpatches
ms_original_l   = [34.2, 57.1, 34.0, 24.6, 26.1, 14.8]
ms_my_det_l     = [39.2, 54.4, 42.5, 50.6, 30.9, 37.0]
ms_my_des_l     = [44.2, 53.7, 39.5, 27.0, 35.0, 20.0]

vgg_ms = 51.3 # ms on hpatch with my vgg variant with pool3 des

plt.figure(1, figsize=(10, 3))
plt.title('Integration of our detector')
G = gridspec.GridSpec(1, 2)

ax1 = plt.subplot(G[0, 0])
ax1.set_ylabel('Matching score')
ax1.set_title('Hpatches: Matching score')

ax1.bar(absc-width/2, ms_original_l, width, color=color_l, edgecolor=ecolor_l)
ax1.bar(absc+width/2, ms_my_det_l, width, color=color_l, edgecolor=ecolor_l, hatch='.')
ax1.bar(absc+3*width/2, ms_my_des_l, width, color=color_l, edgecolor=ecolor_l, hatch='\\')

ax1.bar(0, vgg_ms, width, color=method2color_d['elf-vgg'], edgecolor='black')

for i, ms in enumerate(ms_my_det_l):
    ax1.text(absc[i]+x_shift_right, ms + 1, str(ms), color='black', fontsize=bar_fontsize)
for i, ms in enumerate(ms_original_l):
    ax1.text(absc[i]-x_shift_left, ms + 1, str(ms), color='black', fontsize=bar_fontsize)
for i, ms in enumerate(ms_my_des_l):
    ax1.text(absc[i]+x_shift_right_2, ms + 1, str(ms), color='black', fontsize=bar_fontsize)

ax1.text(-.5, vgg_ms + 1, str(vgg_ms), color='black', fontsize=bar_fontsize)

ax1.set_xticks(np.hstack((np.array([0]), absc)))
ax1.set_xticklabels(['elf-vgg'] + method_l, fontsize=10)
ax1.axis([0-width-0.3, np.max(absc)+2*width+0.3, 0, 75])

##############################################################################
# webcam
ms_original_l   = [18.1, 32.4, 17.8, 10.1, 8.3, 1.3]
ms_my_det_l     = [26.7, 39.6, 30.8, 36.8, 19.1,6.6]
ms_my_des_l     = [30.7, 34.6, 26.8, 13.2, 21.4, 13.9]

vgg_ms = 43.7 # ms on webcam with my vgg variant

ax2 = plt.subplot(G[0, 1])
ax2.set_ylabel('Matching score')
ax2.set_title('Webcam: Matching score')

ax2.bar(absc-width/2, ms_original_l, width, color=color_l, edgecolor=ecolor_l)
ax2.bar(absc+width/2, ms_my_det_l, width, color=color_l, edgecolor=ecolor_l, hatch='.')
ax2.bar(absc+3*width/2, ms_my_des_l, width, color=color_l, edgecolor=ecolor_l, hatch='\\')

ax2.bar(0, vgg_ms, width, color=method2color_d['elf-vgg'], edgecolor='black')

ax2.set_xticks(absc)
ax2.set_xticklabels(method_l, fontsize=10)
ax2.axis([0-width-0.3, np.max(absc)+2*width+0.3, 0, 75])

for i, ms in enumerate(ms_my_det_l):
    ax2.text(absc[i]+x_shift_right, ms + 1, str(ms), color='black', fontsize=bar_fontsize)
for i, ms in enumerate(ms_original_l):
    ax2.text(absc[i]-x_shift_left, ms + 1, str(ms), color='black', fontsize=bar_fontsize)
for i, ms in enumerate(ms_my_des_l):
    ax2.text(absc[i]+x_shift_right_2, ms + 1, str(ms), color='black', fontsize=bar_fontsize)
ax2.text(-.5, vgg_ms + 1, str(vgg_ms), color='black', fontsize=bar_fontsize)

ax2.set_xticks(np.hstack((np.array([0]), absc)))
ax2.set_xticklabels(['elf-vgg'] + method_l, fontsize=10)
ax2.axis([0-width-0.3, np.max(absc)+2*width+0.3, 0, 75])


plt.tight_layout()
plt.savefig('fig/fig9.png')
plt.close()

toto = cv2.imread('fig/fig9.png')
cv2.imshow('fig11', toto)
cv2.waitKey(0)

