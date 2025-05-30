# traction-force-worm
traction force estimation in freely moving worms **C. elegans** on PAA gels

## Installation

Download the package and unpack. Go to its directory.
The software was tested on Windows 11. 
Is it recommended to set up a conda environment:

```conda env create --name your_name --file=environment.yml```

Wait a couple of minutes until it is installed.

## Description
0. **STEP0**
   + prerequisites to apply the pipeline is to have a worm image segmented. I used annotator tracker from [microSAM](https://github.com/computational-cell-analytics/micro-sam) napari plugin, with some fine-tuning (worm should be white and background black and saved as tif file, in sample dataset committed_objects.tif).
1. **STEP1_calculate_offset_piv_tractions.py**
    + This step will save *.pkl file with worm, background and tractions estimated for each frame in the video. This step will usually take a couple of hours.
2. **STEP2_skeletonise.py**
    + This step creates the skeletons from binary segmented videos. This step will usually take around an hour.
3. **STEP3_annotate_skeleton_orientation.py**
    + after this step is important to fill in the flip_skeleton.csv file, according to a given example that include the information if skeleton is oriented
       + head to tail: fill in 0 value in "flip?" column, or
       + tail to head: fill in 1 value in "flip?" column.
4. **STEP4_quantify_tractions.py**
    + This step projects and decomposes the tractions into normal and tangential components along the worm body using the skeleton data. This step will usually take tens of minutes.


## How to cite 
part of the manuscript: Active sensing of substrate mechanics optimizes locomotion of *C. elegans*, A. Pidde, C. Agazzi, M. Porta-de-la-Riva, C. Martínez-Fernández, A. Lorrach, A. Bijalwan, E. Torralba, R. Das, J. Munoz, M. Krieg

sample data can be found under the link https://icfo-my.sharepoint.com/:f:/r/personal/apidde_icfo_net/Documents/codes%20paper/TFM/data_traction_force_worm?csf=1&web=1&e=PBwybE
