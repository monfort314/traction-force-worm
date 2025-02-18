# traction-force-worm
traction force estimation in freely moving worms **C. elegans** on PAA gels

## Description
0. **STEP0**
   + prerequisites to apply the pipeline is to have a worm image segmented. I used annotator tracker from [microSAM](https://github.com/computational-cell-analytics/micro-sam) napari plugin, with some fine-tuning (worm should be white and background black and saved as tif file).
1. **STEP1_calculate_offset_piv_tractions.py**
    + This step will save *.pkl file with worm, background and tractionsestimated for each frame in the video 
2. **STEP2_skeletonise.py**
    + This step creates the skeletons from binary segmented videos. 
3. **STEP3_annotate_skeleton_orientation.py**
    + after this step is important to fill in the flip_skeleton.csv file, according to a given example that include the information if skeleton is oriented
       + head to tail: fill in 0 value in "flip?" column, or
       + tail to head: fill in 1 value in "flip?" column.
4. **STEP4_quantify_tractions.py**
    + This step projects and decomposes the tractions into normal and tangential components along the worm body using the skeleton data.


## How to cite 
part of the manuscript: Active sensing of substrate mechanics optimizes locomotion of *C. elegans*, A. Pidde, C. Agazzi, M. Porta-de-la-Riva, C. Martínez-Fernández, A. Lorrach, A. Bijalwan, E. Torralba, R. Das, J. Munoz, M. Krieg
