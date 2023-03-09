# Cytosponge Triage

This repository contains the functional implementation of the code for analysing histopathology slides produced using a Cytosponge, from the following papers: 

Triage-driven diagnosis of Barrett’s esophagus for early detection of esophageal adenocarcinoma using deep learning - [Paper](https://www.nature.com/articles/s41591-021-01287-9) [Code](https://github.com/markowetzlab/cytosponge-triage)

Use of a Cytosponge biomarker panel to prioritise endoscopic Barrett's oesophagus surveillance: a cross-sectional study followed by a real-world prospective pilot - [Paper](https://doi.org/10.1016/S1470-2045(21)00667-7) [Code](https://github.com/markowetzlab/barretts-progression-detection)

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

<details>
<summary>quality_control.py</summary>

This file interprets the suitability of a given cytosponge slide from a patient. This code takes in a H&E or TFF3 slide and outputs a count for the number of target tiles and secondary tiles as well as an optional annotation file for viewing.

Arguments:
```
--description, takes a String to save the location of results to, defaults to triage

--stain, "he" or "tff3" - Flag to specify the type of data being used
--labels, CSV file containing pathologist ground truth

--network, defaults to VGG 16, specify architecture to use: see models for available
--model_path, path to stored model weights

--slide_path, Path to Slide(s) location/root folder
--format, WSI Extension name, default is .ndpi
--tile_size, Size of tile to generate for model input, default is 400 pixels
--overlap, Fraction of tile edge to extract with overlapping neighboring tiles
--foreground_only, Flag to detect foreground of slide and only perform analysis on tissue
```

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
</details>

***

## Model Training

<details>
<summary>train.py</summary>

</details>

***

## Model Inference
<details>
<summary>inference.py</summary>

</details>

## Model Evaluation
<details>
<summary>evaluate.py</summary>

</details>

## Model Visualisation
<details>
<summary>visualize.py</summary>

</details>

If you use any of the work published in this paper please use the reference below:
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
