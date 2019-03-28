# main.py ---
#
# Filename: main.py
# Description: WRITEME
# Author: Kwang Moo Yi, Lin Chen
# Maintainer: Kwang Moo Yi
# Created: ???
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), EPFL Computer Vision Lab.

# Code:


import os
import sys
import platform

import numpy as np
import tensorflow as tf

import getpass

from config import get_config, save_config

config = None

#import evaluation.cst as cst
import tools.cst as cst


def main(_):

    # Create a random state using the random seed given by the config. This
    # should allow reproducible results.
    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    # Train / Test
    if config.task == "train":
        # Import trainer module
        from trainer import Trainer

        # Create a trainer object
        task = Trainer(config, rng)
        save_config(config.logdir, config)

    else:
        # Import tester module
        if 'hpatch' in cst.DATA:
            from methods.lift.evaluation.tester_hpatch import Tester
            print('WARNING: computation on the hpatch dataset')
          
        
        if cst.DATA=='webcam':
            from methods.lift.evaluation.tester_webcam import Tester
            print('WARNING: computation on the webcam dataset')
           
        if cst.DATA=='strecha':
            from methods.lift.evaluation.tester_strecha import Tester
            print('WARNING: computation on the strecha dataset')


        # Create a tester object
        task = Tester(config, rng)

    # Run the task
    task.run()


if __name__ == "__main__":
    config, unparsed = get_config(sys.argv)
    #print(sys.argv)
    
    # ELF additions
    # ugly addition to the lift parser
    config.elf_kp = 0
    config.kp_dir_id = 0
   
    for l in sys.argv:
        if 'elf_kp' in l:
            v = int(l.split("=")[1])
            config.elf_kp = v

        if 'kp_dir_id' in l:
            config.kp_dir_id=l.split("=")[1]

    print('config.elf_kp', config.elf_kp)
    print('config.kp_dir_id', config.kp_dir_id)

    #if len(unparsed) > 0:
    #    raise RuntimeError("Unknown arguments were given! Check the command line!")

    # Alias to bypass the scratch drive
    # Also if in Canada
    username = getpass.getuser()

    # environment variables are non-portable black magic
    host = platform.node()
    print("User and hostname: {}@{}".format(username, host))

    if "gra" in host or "cedar" in host or "cdr" in host:
        print('Forcing remote folders for Compute Canada nodes'.format(username))
        config.data_dir = "/scratch/{}/Datasets/".format(username)
        config.temp_dir = "/scratch/{}/Temp/".format(username)
        config.scratch_dir = "/scratch/{}/Temp/".format(username)
    elif not config.use_local:
        print('Forcing remote folders for user "{}"'.format(username))
        config.data_dir = "/cvlabdata2/home/{}/Datasets/".format(username)
        config.temp_dir = "/cvlabdata2/home/{}/Temp/".format(username)
        config.scratch_dir = "/cvlabdata2/home/{}/Temp/".format(username)

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

#
# main.py ends here
