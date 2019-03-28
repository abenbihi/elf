# ELF-Det
Appendix code to reproduce the paper results.

General structure: 
- datasets: datasets used in the paper
- docker: tools to build the docker img
- meta: network weights, misc. files
- methods: feature extraction methods
  - cv: opencv implemented ones
  - elf\_superpoint: superpoint variant of elf
  - elf\_vgg: vgg variant of elf
  - superpoint: integration of code released by superpoint authors
  - tilde: integration of c++ code of the tilde authors
- plots: scripts to plot the paper figures
- scripts: examples on how to run the experiments
- tools: metrics code, misc. functions used by elf
   

        git clone --recursive https://github.com/ELF-det/elf.git

# Installation
We recommend using docker to have an image with all the dependencies. 
All the python dependencies are specified in `requirements.txt`. 

Currently, the online display of the image matches is not supported inside
docker (because of the issue with X server). If you run the code inside docker,
we recommend writing the images and their matches to disk to see them.

We require the specified OpenCV versions as other versions may lead to error
messages regarding the patented methods (e.g. sift).

This code requires a GPU.  We are not aware of hardware models requirements.
This code is tested on the following models: QuadroM2200, K20 and GT1080Ti.


##### Direct installation
Install the dependencies specified in `requirements.txt`. 

    sudo pip3 install -r requirements.txt

Warning: if opencv complains about patented algoritms (cf error message at the
bottom of the page), one solution is to install these two specific versions:

    sudo pip3 install opencv-python==3.4.0.14
    sudo pip3 install opencv-contrib-python==3.4.2.17

If you have issues with installing tensorflow, run this command instead:

    sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp35-cp35m-linux_x86_64.whl

##### Docker installation

###### Image compilation
Requires nvidia-docker. Installation instructions can be found at
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
Edit the file `docker/Makefile` to the path where you clone the data.

    EXTERNAL_PATH_TO_WS=/path/to/elf

Build the image
    
    cd docker
    make root

This generates the docker image and run it once the compilation is done. It
mounts the elf directory onto the docker image workspace directory.

It can take some time, so while it builds, get the datasets and the weights.

###### Image running

Once built, you can run the image with 

    nvidia-docker run --volume="/path/to/elf":"/home/ws" -it  -u root elf bash

#### Datasets
Download all the datasets used in the paper with the provided script:

    cd datasets
    chmod +x get_datasets.sh
    ./get_datasets.sh


#### Weights
Download the pre-trained weights of the networks used in paper with the
provided scripts:

    cd meta/weights/
    chmod +x get_weights.sh
    ./get_weights.sh


# Run 

Set the working space path in `tools/cst.py`. If you cloned you directory in
`/path/to/ws` then:

    WS_DIR='/path/to/ws/elf/'
    WS_DIR='/home/ws/' # for docker
    

Set the dataset you want to use in `tools/cst.py'`:

    DATA = data_name

with `data_name` in:
- `'hpatches'`: for the general performances
- `'hpatches_rot'`: for the multi-angle version of hpatches
- `'hpatches_s'`: for the multi-scale version of hpatches
- `'webcam'`: for the webcam dataset
- `'strecha'`: for the strecha dataset

Set `DEBUG` to true if you want to see the matches on the images online.
Only available if you run the code outside of docker or if you can connect the
right docker port that I did not find (sorry).

    DEBUG = (1==1)

## OpenCV methods

To evaluate an opencv method on experiment number `<trials>`:

    ./scripts/bench_cv.sh <method> <data> <trials>
    # e.g.
    ./scripts/bench_cv.sh surf hpatch 0
    ./scripts/bench_cv.sh sift strecha 1
 
with:
- `<method>` in `sift`, `surf`, `orb`, `kaze`, `mser`.
- data in `hpatch` (for hpatches, hpatches\_s, hpatches\_rot), `webcam`, `strecha`.

The results are stored in the file `res/<method>/<trials>/log.txt` with the
following format
    
    *** i_ajuntament *** 0:00
    ** 2 **
    rep:0.568 - N1:500 - N2:500 - M:284
    ms:0.332 - N1:500 - N2:500 - M:284 - M_d:500 - inter:166
    ** 3 **
    rep:0.281 - N1:500 - N2:417 - M:117
    ms:0.065 - N1:500 - N2:417 - M:117 - M_d:417 - inter:27
   
The first line holds the scene name. It is followed by the image id being
matches with the first image of the scene. The two following lines holds the
metrics. 

- rep: repeatability.
- N1/N2: number of keypoints detected in image1/image2.
- M: number of matching detected keypoints according to their geometric
  distance.
- M\_d: number of matching detected keypoints according to their descriptor
  distance.
- inter: intersection of the keypoints matched based on their geometric
  distances and the keypoints matched based on their descriptor distance.

To compute the average repeatability and matching score for that expriments:

    python3 tools/display_res.py --method <method> --trials <trials>
    # e.g.
    python3 tools/display_res.py --method sift --trials 0
    python3 tools/display_res.py --method sift --trials 1
    >>>> rep:    0.51191
    >>>> ms:     0.24603

#### Evaluate the integration of the ELF detector or the proxy descriptor
Open
the script `scripts/bench_cv.sh` and set `ok=1` to activate the script you want
to run: 
- The script suffixed with `des_proxy` runs the OpenCV detector with the proxy
  descriptor.
- The script suffixed with `det_elf` runs the elf detector with the OpenCV
  descriptor.


## ELF
See OpenCV methods section to set the dataset to play with.

##### VGG variant.
Results are written to `res/elf/`.

    ./scripts/bench_elf.sh <trials> <data>
    python3 tools/display_res.py --method <method> --trials <trials>
    
    # e.g.
    ./scripts/bench_elf.sh 0 hpatch
    # set DATA='strecha' in tools/cst.py
    ./scripts/bench_elf.sh 1 strecha
    # set DATA='webcam' in tools/cst.py
    ./scripts/bench_elf.sh 2 webcam
    python3 tools/display_res.py --method elf --trials 2


##### SuperPoint variant
Results are written to `res/superpoint/`.

    ./scripts/bench_elf_superpoint.sh <trials> <data>
    
    # e.g.
    ./scripts/bench_elf_superpoint.sh 0 hpatch
    python3 tools/display_res.py --method superpoint --trials 0



## SuperPoint
Choose the script to run in `scripts/bench_superpoint.sh` by setting `ok=1`
before the script. This shell scripts allows you to run the following
evaluations.

#### Evaluate superpoint 
To run experiment number `<trials>`:

    ./scripts/bench_superpoints.sh <data> <trials>
    # e.g.
    ./scripts/bench_superpoint hpatch 0


#### Evaluate superpoint detector with proxy descriptor.
The script in 2 steps.
The first step runs superpoint detection and saves the keypoint in txt
format in `res/superpoint/<kp_dir_id>`. The second step runs the proxy
descriptor and stores the evaluation in `res/superpoint/<trials>`.

    ./scripts/bench_superpoints.sh <data> <trials> <kp_dir_id>
    # e.g.
    ./scripts/bench_superpoint.sh 3 hpatch 2
    # writes superpoint detector in res/superpoint/2
    # writes evaluation results in res/superpoint/3


#### Evaluate superpoint descriptor with ELF detector.
The script in 2 steps.
The first step runs elf detection and saves the keypoint in txt
format in `res/elf/<kp_dir_id>`. The second step runs the superpoint
descriptor and stores the evaluation in `res/superpoint/<trials>`.

    ./scripts/bench_superpoints.sh <data> <trials> <kp_dir_id>
    ./scripts/bench_superpoint.sh 4 hpatch 1
    # writes superpoint detector in res/elf/1
    # writes evaluation results in res/superpoint/4


## TILDE

#### Installation
You need to compile the C++ version of TILDE.

    cd methods/tilde/c++
    mkdir build
    cd build
    cmake ..
    make

#### Run

    ./scripts/bench_tilde.sh <trials> <data>
    python3 tools/display_res.py --method <method> --trials <trials>
    
    # e.g.
    ./scripts/bench_tilde.sh 0 hpatch



## LIFT
Choose the script to run in `scripts/bench_lift.sh` by setting `ok=1`
before the script. This shell sctrips allows you to run the following
evaluations.

#### Evaluate lift detector and descriptor.
Two models are provided: either the rotation-augmented model or the non augmented
one. 
The script runs in 2 steps. 
First generate the lift keypoints and
descriptors and saves them file with `evaluation/gen_lift.sh`: the outputs
are stored in `res/lift/<lift_data_id>`.
Then compute the metrics with `scripts/bench_lfnet.sh`.
    
    ./evaluation/gen_lift.sh <lift_data_id>
    ./scripts/bench_lift.sh <lift_data_id> <data> <trials>
    python3 evaluation/display_res.py --method lift --trials <trials>
    
    # e.g.
    # generate lift keypoint and descriptor in res/lift/0
    ./evaluation/gen_lift.sh 0

    # evaluate lift output in res/lift/0 and writes evaluation in res/lift/1
    ./scripts/bench_lift.sh 0 hpatch 1
    python3 evaluation/display_res.py --method lift --trials 1


#### Evaluate lift detector with proxy descriptor.
First generate the lift keypoints (see previous section): the outputs
are stored in `res/lift/<lift_data_id>`.
Then run the proxy
descriptor and stores the evaluation in `res/elf/<trials>`.

    ./scripts/gen_lift.sh <lift_data_id>
    ./scripts/bench_lift.sh <lift_data_id> <data> <trials> 

    # e.g.
    # generate lift keypoint and descriptor in res/lift/0
    ./sriptsn/gen_lift.sh 0

    # evaluates proxy desriptor with lift kp stored res/lift/0 
    # writes evaluation in res/lift/1
    ./scripts/bench_lift.sh 0 hpatch 1
    python3 tools/display_res.py --method lift --trials 1


#### Evaluate lift descriptor with ELF detector 
The script in 3 steps.
- The first step runs elf detection and saves the keypoint in txt format in
`res/elf/<kp_dir_id>`. 
- The second step generates the lift descriptors and
stores the output in `/res/lift/<lift_data_id>`. 
- The final step computes the
metrics and stores the evaluation in `res/superpoint/<trials>`.

        ./scripts/bench_lift.sh <lift_data_id> <data> <trials> <kp_dir_id>
        
        # e.g.
        ./scripts/bench_lift.sh 2 hpatch 3 0
        # writes elf keypoints in res/elf/0
        # writes lift descriptor in res/lift/2
        # write evaluation in res/lift/3
        python3 tools/display_res.py --method lift --trials 3



## LF-Net
Choose the script to run in `scripts/bench_lfnet.sh` by setting `ok=1`
before the script. This shell sctrips allows you to run the following
evaluations.

#### Evaluate lfnet detector and descriptor
With either the indoor or the outdoor model.

    ./scripts/bench_lfnet.sh <trials> <data> <model>
    python3 tools/display_res.py --method <method> --trials <trials>
    
    # e.g.
    ./scripts/bench_lfnet.sh 0 hpatch indoor
    ./scripts/bench_lfnet.sh 1 hpatch outdoor
    python3 tools/display_res.py --method lnet --trials 0


#### Evaluate lfnet detector with proxy descriptor.
The script in 2 steps.
The first step runs lfnet detection and saves the keypoint in txt
format in `res/lfnet/<kp_dir_id>`. The second step runs the proxy
descriptor and stores the evaluation in `res/elf/<trials>`.

    ./scripts/bench_lfnet.sh <data> <trials> <kp_dir_id>
    ./scripts/bench_lfnet.sh 3 hpatch indoor 2
    # writes lfnet detector in res/lfnet/2
    # writes evaluation results in res/lfnet/3


#### Evaluate lfnet descriptor with ELF detector.
The script in 2 steps.
The first step runs elf detection and saves the keypoint in txt
format in `res/elf/<kp_dir_id>`. The second step runs the lfnet
descriptor and stores the evaluation in `res/lfnet/<trials>`.

    ./scripts/bench_lfnet.sh <data> <trials> <kp_dir_id>
    ./scripts/bench_lfnet.sh 4 hpatch indoor 1
    # writes lfnet detector in res/elf/1
    # writes evaluation results in res/lfnet/4

### ELF-LF-Net
    
    ./scripts/bench_elf_lfnet.sh <trials>

    python3 tools/display_res.py --method elf-lfnet --trials <trials>
    python3 tools/display_res.py --method elf-lfnet --trials 0



# Plots
Each script plot is named after the paper figure number.
    
    cd plots
    python3 fig5_hpatch.py 

# Known issues

## OpenCV versions
- Pb: OpenCV complains that sift is in the non-free package of opencv

        C:\projects\opencv-python\opencv_contrib\modules\xfeatures2d\src\sift.cpp:1207:
        error: (-213:The function/feature is not implemented) 
        This algorithm is patented and is excluded in this configuration; Set
        OPENCV_ENABLE_NONFREE CMake 
        option and rebuild the library in function
        'cv::xfeatures2d::SIFT::create'

- Sol: Please, install the specific required versions above as there are some
  versions imcompability issues. I don't know why, nor how to solve it so the
  simple solution is to use the given versions that are tested and work.



