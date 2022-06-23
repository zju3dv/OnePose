# OnePose Demo on Custom Data (WIP)
In this tutorial we introduce the demo of OnePose running with data captured
with our **OnePose Cap** application available for iOS device. 
The app is still under preparing for release.
However, you can try it with the [sample data]() and skip the first step.  

### Step 1: Capture the mapping sequence and the test sequence with OnePose Cap. 
#### The app is under brewing🍺 coming soon.

### Step 2: Organize the file structure of collected sequences
1. Export the collected mapping sequence and the test sequence to the PC.
2. Rename the **annotate** and **test** sequences directories to ``your_obj_name-annotate`` and `your_obj_name-test` respectively and organize the data as the follow structure:
    ```
    |--- /your/path/to/scanned_data
    |       |--- your_obj_name
    |       |       |---your_obj_name-annotate
    |       |       |---your_obj_name-test
    ```
   Refer to the [sample data]() as an example.
3. Link the collected data to the project directory
    ```shell
    REPO_ROOT=/path/to/OnePose
    ln -s /path/to/scanned_data $REPO_ROOT/data/demo
    ```
   
Now the data is prepared!

### Step 3: Run OnePose with collected data
Download the [pretrained OnePose model](https://drive.google.com/drive/folders/1VjLLjJ9oxjKV5Xy3Aty0uQUVwyEhgtIE?usp=sharing) and move it to `${REPO_ROOT}/data/model/checkpoints/onepose/GATsSPG.ckpt`.

[Optional] To run OnePose with tracking modeule, pelase install [DeepLM](https://github.com/hjwdzh/DeepLM.git).
Please make sure the sample program in `DeepLM` can be correctly executed to ensure successful installation.


Execute the following commands, and a demo video naming `demo_video.mp4` will be saved in the folder of the test sequence.
```shell
REPO_ROOT=/path/to/OnePose
OBJ_NAME=your_obj_name

cd $REPO_ROOT
conda activate OnePose

bash scripts/demo_pipeline.sh $OBJ_NAME

# [Optional] running OnePose with tracking
export PYTHONPATH=$PYTHONPATH:/path/to/DeepLM/build
export TORCH_USE_RTLD_GLOBAL=YES

bash scripts/demo_pipeline.sh $OBJ_NAME --WITH_TRACKING 

```
