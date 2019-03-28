"""
========================
Visualizing named colors
========================

Simple plot example with the named colors and its visual representation.
"""
from __future__ import division

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import cv2


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]


method2color_d = {}
method2color_d['elf-vgg']       = colors['blue'] # C1
method2color_d['vgg_sal']       = colors['blue'] # C1
method2color_d['vgg']           = colors['blue'] # C1
method2color_d['lfnet']         = colors['orange'] #C2
method2color_d['superpoint']    = colors['green'] #C3
method2color_d['SP']            = colors['green'] #C3
method2color_d['lift']          = colors['red'] #C4
method2color_d['sift']          = colors['purple'] 
method2color_d['surf']          = colors['brown']
method2color_d['orb']           = colors['pink']
method2color_d['kaze']          = colors['gray']
method2color_d['alex_sal']      = colors['olive'] 
method2color_d['xcep_sal']      = colors['darkred'] # C9
method2color_d['tilde']         = colors['crimson'] 
method2color_d['mser']          = colors['lime' ]
method2color_d['lfnet_sal']     = colors['orchid'] 
method2color_d['sp_sal']        = colors['darkcyan'] 
method2color_d['sobel']         = colors['midnightblue'] 
method2color_d['lapl']          = colors['hotpink'] 



if __name__=='__main__':

    method2color_d = {}
    method2color_d['ELF-VGG']       = colors['blue'] # C1
    method2color_d['LFNet']        = colors['orange'] #C2
    method2color_d['SuperPoint (SP)']    = colors['green'] #C3
    method2color_d['Lift']          = colors['red'] #C4
    method2color_d['SIFT']          = colors['purple'] 
    method2color_d['SURF']          = colors['brown']
    method2color_d['ORB']           = colors['pink']
    method2color_d['KAZE']          = colors['gray']
    method2color_d['ELF-AlexNet']   = colors['olive'] 
    method2color_d['ELF-Xception']  = colors['darkred'] # C9
    method2color_d['Tilde']         = colors['crimson'] 
    method2color_d['MSER']          = colors['lime' ]
    method2color_d['ELF-LFNet']   = colors['orchid'] 
    method2color_d['ELF-SP']        = colors['darkcyan'] 
    method2color_d['Sobel']         = colors['midnightblue'] 
    method2color_d['Laplacian']     = colors['hotpink'] 
    
    method_l = [
            'LFNet', 'SuperPoint (SP)', 'Lift', 'SIFT', 'SURF', 'ORB', 'KAZE', 'Tilde', 'MSER',
            'ELF-VGG', 'ELF-AlexNet', 'ELF-Xception', 'ELF-LFNet', 'ELF-SP', 'Sobel', 'Laplacian']

   
    n = len(method2color_d.values())
    print(n)
    #ncols = 9
    #nrows = n // ncols + 1
    #fig, ax = plt.subplots(figsize=(14, 1))
    
    ncols = 4
    nrows = n // ncols
    fig, ax = plt.subplots(figsize=(14, 3))

    # Get height and width
    X, Y = fig.get_dpi() * fig.get_size_inches()
    h = Y / (nrows + 1)
    w = X / ncols
    
    for i, method in enumerate(method_l):
        col = i % ncols
        row = i // ncols
        y = Y - (row * h) - h
    
        xi_line = w * (col + 0.05)
        xf_line = w * (col + 0.25)
        xi_text = w * (col + 0.3)
    
        ax.text(xi_text, y, method, fontsize=(h * 0.4),
                horizontalalignment='left',
                verticalalignment='center')
        
        print(method2color_d[method])
        ax.hlines(y + h * 0.1, xi_line, xf_line,
                  color=method2color_d[method], linewidth=(h * 0.6))
    
    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()
    
    fig.subplots_adjust(left=0, right=1,
                        top=1, bottom=0,
                        hspace=0, wspace=0)
    #plt.show()
    plt.savefig('res/paper/method2legend.png')
    plt.close()
    
    toto = cv2.imread('res/paper/method2legend.png')
    cv2.imshow('toto', toto)
    cv2.waitKey(0)


