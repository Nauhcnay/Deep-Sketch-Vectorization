# DeepSketch
This is the code repository associated with the paper [Deep Sketch Vectorization via Implicit Surface Extraction](https://cragl.cs.gmu.edu/sketchvector/) by Chuan Yan, Yong Li, Deepali Aneja, Matthew Fisher, Edgar Simo-Serra and Yotam Gingold from SIGGRAPH 2024.

## Install dependencies

It is recommended to first use a virtual environment, then install the following packages

```
pytorch
opencv
svgpathtools
tqdm
scikit-learn
rdp
scikit-image
matplotlib

## these modules below can only be installed via pip ##
aabbtree
edge_distance_aabb
ndjson
eel
```

It could be either installed via **pip**:
```
python3 -m venv deepsketch
source deepsketch/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install opencv svgpathtools tqdm scikit-learn rdp scikit-image matplotlib aabbtree edge_distance_aabb ndjson eel
```

or via **conda**:

```
conda create -n deepsketch python
conda activate deepsketch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge opencv svgpathtools tqdm scikit-learn rdp scikit-image matplotlib
pip install aabbtree edge_distance_aabb ndjson eel
```
Then also clone this repository via:
```
git clone https://github.com/Nauhcnay/Deep-Sketch-Vectorization
cd Deep-Sketch-Vectorization
```
## Download pretrained models
Assume you are at the project root, you can download all needed model via:
```
git clone https://huggingface.co/waterheater/deepsketch pretrained
```
There are two sets of pretrained models:
1. `ndc_full.pth` and `udf_full.pth` are the full size model which requires 8GB or above VRAM to run, these models will provide the best vectorization performance.
2. `ndc_light.pth` and `udf_light.pth` are the light size model which only needs less than 4GB VRAM to run.

And if you wish to train your own model, please refer to section [Training](#training)

We also added Edgar's [Line Thinning](https://github.com/bobbens/line_thinning) and Xiaoyu Xiang's [Anime2Sketch](https://github.com/Mukosame/Anime2Sketch/) as proprocess options in our pipeline.

## Vectorize single image
We recommand to use our webUI, it provides full access to all features and additional interactive refinement support duruing vectorization.
To start, simply run
```
python server.py
```
and then open your browser to address:
```
http://localhost:8888/main.html
```
after you finalized your vectorization, you will find the vectorization results in 
```
./web/output
```
A full instruction of how to using our webUI could be found in here (todo: create another readme for webUI usage)

## Vectorize images in a folder
Assume we are at the project root, run
```
python predict_s1.py --input path_to_input_folder --output path_to_output_folder --refine --rdp  
```
To see the full options, you can see it by 
```
python predict_s1.py --help
```

## Training
### Download the created full training dataset
If you don't want to create the training set from scratch, you can download the one that is created by us:
```
mkdir data
cd ./data
wget https://huggingface.co/datasets/waterheater/deepsketch-dataset/resolve/main/deepsketch_dataset_full.zip
unzip deepsketch_dataset_full.zip
```
Then jump to section [Start training](#start-training) directly.

Please noted that the zip file is 17.3GB and its unzipped size will be around 66GB. 

### Download source dataset and sampling
Download the [Quick Draw!](https://github.com/googlecreativelab/quickdraw-dataset#preprocessed-dataset) dataset (Simplified Drawing files) and the [Creative Creature sketch Dataset](https://drive.google.com/drive/u/5/folders/14ZywlSE-khagmSz23KKFbLCQLoMOxPzl). Unzip them into `./data/quick_draw` and `./data/creative`, respectively. 
run:

```
cd ./data
python preprocess.py
```

This will create two folders `full` and `sample`. The first one contains randomly sampled 100K sketches from the two datasets above, all sketch will be converted and saved as SVG file. Please use ```full``` as the folder for the following training set creation steps if you want to train a usable model to the end.

The second folder only contains 10K sketches randomly sampled from the full training folder, and is only used for quick network debug, it won't gives you a generalized model.

### Create sketch keypoint ground truth
```
mkdir ./full/keypt
python junction_detection.py ./full/svg -o /full/keypt -x
```
This will create the ground truth of sketch key points.

### Sample sketch centerline (Unsigned Distance Field) and generate the Dual Contouring edge flags
```
mkdir ./full/gt
cd ../utils
python ndc_tools.py
```
This will first sample the UDF from each vector sketch and then combine with the keypoint ground truth to generate the final ground truth for training.

### Rasterize and stylize the vector sketch
Use vscode with [ExtendScript Debugger](https://marketplace.visualstudio.com/items?itemName=Adobe.extendscript-debug) to run the script `./dataset/AI_add_brushes.jsx`.
We have 7 different brush style so you will need to run this script 7 times to generate all input raster sketches. Before you run this script make sure to update the code at line 441th:
```
// 0: basic, 
// 1: Calligraphic Brush 1, 
// 2: BrushPen 42, 
// 3: Charcoal_smudged_3, 
// 4: HEJ_TRUE_GRIS_M_STROKE_04, 
// 5: BrushPen 111, 
// 6: Comic Book_Contrast 3
// update this variable to select brush type above before you run this script everytime
var random = 0; 
```
For more help of how to use the ExtendScript Debugger, here is [a nice introduction](https://github.com/ivanpuhachov/line-drawing-vectorization-polyvector-flow-dataset)

### Start training
Assume we are at the project root, you can train the similar models as in our paper by using the following full commands, respectively.
#### S1: Train the Distance Field Prediction network

```
python train_s1.py --bs 16 --workers 16 --dist_mode l1 --lr 5e-5 --up_scale --dist_clip 4.5 --paper_background --eval --jpg --hourglass_input_channels 128 --hourglass_channels 256 --cardinality
```

#### S2: Train the Line Reconstruction network

```
python train_s2.py --bs 20 --bn --keypt --nl 0 --review --epoch 200 --focal --workers 16 --dist_clip 4.5 --skel --mgs --msb
```

#### S3: Train the Full Deep Sketch Vectorization pipline
After you have compelted the **S1** and **S2**, put your trained models into ./pretrained. Let's say they are name as "**udf.pth**" from S1 and "**ndc.pth**" from S2, repsectively. Run:

```
python train_s1.py --ndc_add --udf udf.pth --ndc ndc.pth --workers 16 --bs 8 --review_all
```
If you have account on [Weights & Biases](https://wandb.ai/site/), you can also add parameters: ```--usr your_wandb_user_name```, ```--name the_training_task_name``` and ``` --log ``` at each stage of your traning, this will log most the training details to wandb.

Also if you found the code can't strat the training task normally, you can just try 
```
python train_s1.py --deubg
```
and
```
python train_s2.py --deubg
```
This will start a training with batch size 1. It will output more training details and stops just after 1 iteration, which could help you to have a quick debug for the issue.
