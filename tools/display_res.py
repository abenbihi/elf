"""
python tools/display_res_rep.py --method orb --metric ms --trials 11
"""

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--trials', required=True, type=str, default='0', help='xp id')
parser.add_argument('--method', required=True, type=str, default='sift')
args = parser.parse_args()


res_dir = os.path.join('res', args.method, args.trials)
log_fn = os.path.join(res_dir, 'log.txt')
log_l = [l.split("\n")[0] for l in open(log_fn).readlines() ]

rep, ms = [],[]
for l in log_l:
    if 'rep' in l:
        rep.append(float(l.split("-")[0].split(":")[1]))

    if 'ms' in l:
        ms.append(float(l.split("-")[0].split(":")[1]))

rep = np.array(rep)
ms = np.array(ms)
print('rep:\t%.5f'%np.mean(rep))
print('ms:\t%.5f'%np.mean(ms))


