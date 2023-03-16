# Cytosponge Triage

This repository contains the functional implementation of the code for analysing histopathology slides produced using a Cytosponge, from the following papers: 

Triage-driven diagnosis of Barrett’s esophagus for early detection of esophageal adenocarcinoma using deep learning - [Paper](https://www.nature.com/articles/s41591-021-01287-9) [Code](https://github.com/markowetzlab/cytosponge-triage)

Use of a Cytosponge biomarker panel to prioritise endoscopic Barrett's oesophagus surveillance: a cross-sectional study followed by a real-world prospective pilot - [Paper](https://doi.org/10.1016/S1470-2045(21)00667-7) [Code](https://github.com/markowetzlab/barretts-progression-detection)

Trained model weights can be found at the following link: [Weights](https://drive.google.com/drive/folders/1XYv1OdUg_z_t0GXq2k2a9hhSxlTXvxuZ?usp=sharing)

## Requirements
<details>
<summary> Cloning the Repository </summary>

To copy this repository into your local workspace you can copy one of the functions using the green Code button at the top of this page or alternatively copy the below command:
```
git clone https://github.com/markowetzlab/slide-classifier.git
```
</details>

<details>
<summary>Virtual Environment</summary>
To use this software, it is recommended you use a virtual or conda environment. 

For a virtual environment, you can follow the below intructions using virtualenv and requirements.txt.
```
virtualenv -p python3 <name of env>
source <name of env>/bin/activate
pip install -r requirements.txt
```

Alternatively, you can first install anaconda and create a virtual environment using the below commands:
```
conda create -y --name <name of env>
conda activate <name of env>
conda install -c conda-forge --file conda_requirements.txt
```
Finally, as Slidl is not available through conda:
```
pip install slidl
```
</details>

***

## Quality Control:

This file interprets the suitability of a given cytosponge slide from a patient. This code takes in a H&E or TFF3 slide and outputs a count for the number of target tiles and secondary tiles as well as an optional annotation file for viewing.

By default, this file uses the VGG-16 network, which is trained separately to perform quality control analysis on H&E slides (i.e. Gastric-columnar Epithelium and Intestinal Metaplasia detection), or Goblet cell detection from TFF3 slides, with thresholds determined from the paper.

<details>
<summary>quality_control.py</summary>

Arguments:
```
--description, takes a String to save the location of results to, defaults to triage

Slide properties:
--stain, choices are "he" or "tff3" - Flag to specify the type of data being used
--slide_path, Path to Slide(s) location/root folder
--format, WSI Extension name, default is .ndpi
--tile_size, Size of tile to generate for model input, default is 400 pixels
--overlap, Fraction of tile edge to extract with overlapping neighboring tiles
--foreground_only, Flag to detect foreground of slide and only perform analysis on tissue
--labels, CSV file containing pathologist ground truth

Model and path to model weights
--network, defaults to VGG 16, specify architecture to use: see models for available
--model_path, path to stored model weights, must specify

Data prepocessing:
--channel_means, Channel Means as a list to normalise around, default is the imagenet channel averages i.e. [0.485, 0.456, 0.406]
--channel_stds, Channel standard deviation to normalise around, default is the imagenet channel std i.e. [0.229, 0.224, 0.225]
--batch_size, Batch size to infer on, defaults to architecture determined batch size
--num_workers, Number of CPU workers

Thresholds:
--qc_threshold, Threshold of model output to consider as positive for target classes, default is 0.99 as determined by the paper
--tff3_threshold, Threshold of model output to consider as positive for target classes, default is 0.93 as determined by the paper
--tile_cutoff, Threhsold number of tiles to consider as positive, default is 6 as determined in the paper

Specify script outputs:
--output, Path to save outputs to
--csv, Flag to save data as CSV file including tile counts
--stats, Flag to produce Precision-Recall plot
--xml, Flag to produce model outputs as annotation files in .xml (ASAP) format
--json, Flag to produce model outputs as annotation files in .geojson (QuPath) format
--vis, Flag to dsiplay the output of the model as a heatmap of areas to analyse
--thumbnail, Flag to save the vis thumbnail, vis must also be active
```
</details>

If you use this code please cite the original paper using the citation below:
```
@article{gehrung2021triage,
  title={Triage-driven diagnosis of Barrett’s esophagus for early detection of esophageal adenocarcinoma using deep learning},
  author={Gehrung, Marcel and Crispin-Ortuzar, Mireia and Berman, Adam G and O’Donovan, Maria and Fitzgerald, Rebecca C and Markowetz, Florian},
  journal={Nature medicine},
  volume={27},
  number={5},
  pages={833--841},
  year={2021},
  publisher={Nature Publishing Group US New York}
}
```
***
## Triage and Automated Diagnosis
The below scripts are used to determine the operating and triage thresholds for the trained model on a given dataset.

First use the inference.py script to generate the model files and associated prediction data for the given slides. These are then used to determine the model accuracy using the evaluate.py script or use the visualise.py script to output the model annotations in a format which can be viewed in ASAP (.xml) or QuPath (.geojson). 

<details>
<summary>inference.py</summary>

Script to perform model diagnosis for a given H&E slide or P53 slide, which can be a single slide or multiple slides. 

Takes a model architecture and path to associated model weights and determines the optimal threshold for considering the cutoff for Atypia in H&E, or aberrant positive columnar tissue in P53 slides.

Outputs a proposed model threshold to achieve best results vs pathologist reviews using AUC and AUPRC.

Arguments:
```
--dataset, Flag to change behaviour depending on the dataset, determines whether to handle consider postive control tissue (delta) and mapping of labels from Y/N to 1/0.

Slide properties:
--stain, choices are "he" or "p53" - Flag to specify the type of data being used
--slide_path, Path to Slide(s) location/root folder
--format, WSI Extension name, default is .ndpi
--tile_size, Size of tile to generate for model input, default is 400 pixels
--overlap, Fraction of tile edge to extract with overlapping neighboring tiles
--foreground_only, Flag to detect foreground of slide and only perform analysis on tissue
--labels, CSV file containing pathologist ground truth

Model and path to model weights
--network, defaults to VGG 16, specify architecture to use: see models for available
--model_path, path to stored model weights, must specify

Atypia classes to consider (i.e. H&E slides):
--dysplasia_separate, Flag whether to separate the atypia of uncertain significance and dysplasia
--respiratory_separate, Flag whether to separate the respiratory mucosa cilia and respiratory mucosa
--gastric_separate, Flag whether to separate the tickled up columnar and gastric cardia
--atypia_separate, lag whether to perform the following class split: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other

P53 classes to consider (i.e. P53 slides):
--p53_separate, Flag whether to perform the following class split: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar

Data prepocessing:
--channel_means, Channel Means as a list to normalise around, default is the BEST2 channel averages i.e. [0.7747305964175918, 0.7421753839460998, 0.7307385516144509],
--channel_stds, Channel standard deviation to normalise around, default is the BEST2 channel std i.e. [0.2105364799974944, 0.2123423033814637, 0.20617556948731974]
--batch_size, Batch size to infer on, defaults to architecture determined batch size
--num_workers, Number of CPU workers

Specify script outputs:
--output, Path to save outputs to, default is results
--csv, Flag to save data as CSV file including tile counts
--stats, Flag to produce associated statistical data
--vis, Flag to dsiplay the output of the model as a heatmap of areas to analyse
--thumbnail, Flag to save the vis thumbnail, vis must also be active
```
</details>

<details>
<summary>evaluate.py</summary>

This script generates the model's results from P53 and H&E slides for a given threshold.

This can generate results for either H&E or P53 separately, but if both are provided then also triages patients based on collective traige analysis.

Arguments:
```
General
--description, Descriptive string to save results to, defaults to triage
--format, suffix extension of whole slide images, default is .ndpi

Labels:
--from-file, Generate stats from existing csv file, takes path to existing data
--labels, CSV file containing slide-level pathologist ground truth to compare against

Atypia classes to consider (i.e. H&E slides):
--dysplasia_separate, Flag whether to separate the atypia of uncertain significance and dysplasia
--respiratory_separate, Flag whether to separate the respiratory mucosa cilia and respiratory mucosa
--gastric_separate, Flag whether to separate the tickled up columnar and gastric cardia
--atypia_separate, lag whether to perform the following class split: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other

P53 classes to consider (i.e. P53 slides):
--p53_separate, Flag whether to perform the following class split: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar

H&E arguments
--he_path, slide root folder for H&E images
--he_inference, path to directory containing H&E inference file(s)
--he_threshold, A threshold above or equal to target tiles (atypia tiles for H&E, aberrant P53 columnar for P53), default is 0.99

P53 arguments
--p53_path, slide root folder for P53 images
--p53_inference, path to directory containing P53 inference file(s)
--p53_threshold, A threshold above or equal to target tiles (atypia tiles for H&E, aberrant P53 columnar for P53), default is 0.99
--control, CSV containing control tissue location.

Ouptput arguments
--output, output path to folder where inference maps will be stored, defaults to results
--csv, Flag to output results as csv
```
</details>

<details>
<summary>visualize.py</summary>
Generate annotation files from model inference files, which can be viewable in ASAP (.xml) format or QuPath (.geojson) format.

Arguments:
```
General
--description, Description, string to save results to.
--stain, he or p53
--labels, file containing slide-level ground truth to use
--target, Target class to identify, if None then defaults to stain class.
--gt, Column containing ground truth labels

Input
--slide_path, slides root folder
--format, Extension of whole slide images, default is .ndpi
--inference, path to directory containing inference file(s)
	
Atypia classes to consider (i.e. H&E slides):
--dysplasia_separate, Flag whether to separate the atypia of uncertain significance and dysplasia
--respiratory_separate, Flag whether to separate the respiratory mucosa cilia and respiratory mucosa
--gastric_separate, Flag whether to separate the tickled up columnar and gastric cardia
--atypia_separate, lag whether to perform the following class split: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other

P53 classes to consider (i.e. P53 slides):
--p53_separate, Flag whether to perform the following class split: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar

Thresholds
--triage_threshold, A lower threshold for target classes (atypia tiles for H&E, aberrant P53 columnar for P53) for triage, default is not to extract these tiles and 0.99
--automated_threshold, Automated threshold with high confidence to mark for the annotation file, default is 0.999

Outputs
--output, path to folder where inference maps will be stored
--xml, Flag to output annotation file in ASAP .xml format
--json, Flag to output annotation file in QuPath .geojson format
--tiles, Save tile images (as .jpg)
--vis, Flag to save thumbnail as heatmaps

```
</details>

If you use any of the work published in this paper please consider citing us using the reference below:
```
@article{pilonis2022use,
  title={Use of a Cytosponge biomarker panel to prioritise endoscopic Barrett's oesophagus surveillance: a cross-sectional study followed by a real-world prospective pilot},
  author={Pilonis, Nastazja Dagny and Killcoyne, Sarah and Tan, W Keith and O'Donovan, Maria and Malhotra, Shalini and Tripathi, Monika and Miremadi, Ahmad and Debiram-Beecham, Irene and Evans, Tara and Phillips, Rosemary and others},
  journal={The Lancet Oncology},
  volume={23},
  number={2},
  pages={270--278},
  year={2022},
  publisher={Elsevier}
}
```
