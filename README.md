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

```
</details>

***

## Quality Control:

This file interprets the suitability of a given cytosponge slide from a patient. This code takes in a H&E or TFF3 slide and outputs a count for the number of target tiles and secondary tiles as well as an optional annotation file for viewing.

By default, this file uses the VGG-16 network, which is trained separately to perform quality control analysis on H&E slides (i.e. Gastric-columnar Epithelium and Intestinal Metaplasia detection), or Goblet cell detection from TFF3 slides, with thresholds determined from the paper.

<details>
<summary>quality_control.py</summary>

<!-- Arguments:
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
--thumbnail, Flag to save the vis thumbnail, vis must also be active -->
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
<details>
<summary>diagnosis.py</summary>

<!-- Arguments:
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
--tile_cutoff, Threshold number of tiles to consider as positive, default is 6 as determined in the paper

Atypia classes to consider (i.e. H&E slides):
--dysplasia_separate, Flag whether to separate the atypia of uncertain significance and dysplasia
--respiratory_separate, Flag whether to separate the respiratory mucosa cilia and respiratory mucosa
--gastric_separate, Flag whether to separate the tickled up columnar and gastric cardia
--atypia_separate, lag whether to perform the following class split: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other

Specify script outputs:
--output, Path to save outputs to
--csv, Flag to save data as CSV file including tile counts
--stats, Flag to produce Precision-Recall plot
--xml, Flag to produce model outputs as annotation files in .xml (ASAP) format
--json, Flag to produce model outputs as annotation files in .geojson (QuPath) format
--vis, Flag to dsiplay the output of the model as a heatmap of areas to analyse
--thumbnail, Flag to save the vis thumbnail, vis must also be active -->
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
