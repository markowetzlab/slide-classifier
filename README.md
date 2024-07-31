# Cytosponge Triage

This repository contains the functional implementation of the code for analysing histopathology slides produced using a EndoSign, from the following papers: 

Triage-driven diagnosis of Barrett’s esophagus for early detection of esophageal adenocarcinoma using deep learning - [Paper](https://www.nature.com/articles/s41591-021-01287-9) [Code](https://github.com/markowetzlab/cytosponge-triage)

Use of a Cytosponge biomarker panel to prioritise endoscopic Barrett's oesophagus surveillance: a cross-sectional study followed by a real-world prospective pilot - [Paper](https://doi.org/10.1016/S1470-2045(21)00667-7) [Code](https://github.com/markowetzlab/barretts-progression-detection)

Trained model weights can be found at the following link: [Weights](https://drive.google.com/drive/folders/1XYv1OdUg_z_t0GXq2k2a9hhSxlTXvxuZ?usp=sharing)

Some paths and outputs of this repository are hard-coded and is not intended for wider use cases. Therefore, you may want to change your paths accordingly.

## Requirements
<details>
<summary> Cloning the Repository </summary>

To copy this repository into your local workspace you can copy one of the functions using the green Code button at the top of this page or alternatively copy the below command:
```
git clone https://github.com/markowetzlab/slide-classifier.git
```

or using the [github cli](https://cli.github.com/) (recommended):
```
gh repo clone markowetzlab/slide-classifier
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
conda create --name <env> --file requirements.txt
conda active <env>
```
</details>

## Patching:
For speed, efficiency, and reproducibility, slides are first segmented using modified code from [CLAM](https://github.com/mahmoodlab/CLAM/tree/master) (Lu et al. 2022) except with our improved Tissue Segmentation algorithm [TissueTector](https://github.com/markowetzlab/tissue_segmentation).

This also allows us to automatically remove pieces of control tissue that are present in IHC slides, including P53 and TFF3.

<details>
<summary>patching.py</summary>
Example usage of this file for H&E images is as follows, which transforms the slide into the HED colour space first and segments using the Eosin colour channel, with optional size and level arguments to segment:

```
python patching.py --source <slide_dir> --save_dir <patch_dir> --patch_size 400 --patch_level 0 --step_size 400 --seg --patch --stitch
```

For IHC images, this script can be modified to remove certain pieces of tissue, for example filtering for only the largest bounding boxes, and segmenting via a grayscale colour space.

You can also specify a list of slides to process with either a txt file or csv file (assuming the slide column is labelled as slide_id).

```
python patching.py --source <slide_dir> --save_dir <patch_dir> --patch_size 400 --patch_level 0 --step_size 400 --seg --patch --stitch --max_bbox 2 --process_list slides.txt
```

A fill list of arguments can be found by runningL
```
python patching.py --help
```

</details>

## Inference:

This all in one file performs inference using the model as defined by the ```stain``` argument interprets the suitability of a given cytosponge slide from a patient. 

This code takes in slide and outputs a count for the number of target class tiles, as well as optional annotation files for viewing in various viewers.

A full list of arguments can be found by running:

```
python inference.py --help
```
This file outputs a .csv file with model outputs for each patch as created in ```patching.py``` for later thresholding.

Optional outputs for this script include:
```
--json geoJSON file to be read in by open-source slide viewer QuPath
--xml annotation file to be read by the ASAP slide viewer software
--ndpa annotation file to be read by Hamamatsu specific NDP viewer software
--images Produce the top-*k* positive tiles of the ranked_class as images for quick viewing
```

<details>
<summary>Quality Control</summary>

Example usage of the inference script for determining the presence of Gastric cardia from the EndoSign:
```
CUDA_VISIBLE_DEVICES=0 python inference.py --stain qc \\
--network vgg16 \\
--model_path <path to qc model (he vgg16)> \\
--slide_path <path to raw slide files to read from> \\
--patch_path <path to hdf5 (.h5) patches from patching.py> \\
--save_dir <path to output folder> \\
--json \\
--process_list <txt or csv files of slides to infer>   
```

By default, this file uses the VGG-16 network, which is trained separately to perform quality control by detecting Gastric-columnar Epithelium from H&E slides, but only supports inference through one GPU.
</details>

<details>
<summary>Positive TFF3 detection</summary>

Example usage of the inference script for determining the presence of positively stained TFF3 goblet cells from the IHC slides:
```
CUDA_VISIBLE_DEVICES=0 python inference.py --stain tff3 \\
--network vgg16 \\
--model_path <path to tff3 model (tff3 vgg16)> \\
--slide_path <path to raw slide files to read from> \\
--patch_path <path to hdf5 (.h5) patches from patching.py> \\
--save_dir <path to output folder> \\
--json \\
--process_list <txt or csv files of slides to infer>   
```
</details>

<details>
<summary>Atypical Gastric Cardia detection</summary>
Example usage of the inference script for determining the early presence of atypical gastric cardia from H&E images:

```
CUDA_VISIBLE_DEVICES=0,1 python inference.py --stain he \\
--network vit_l_16 \\
--model_path <path to atypia model (he vit_l_16)> \\
--slide_path <path to raw slide files to read from> \\
--patch_path <path to hdf5 (.h5) patches from patching.py> \\
--save_dir <path to output folder> \\
--json \\
--process_list <txt or csv files of slides to infer>   
```
This model supports the use of multi-gpu processing by specifying the GPU IDs in CUDA_VISIBLE_DEVICES

</details>
<details>
<summary>Positive P53 detection</summary>
Example usage of the inference script for determining the presence of positive p53 stained nuclei from P53-stained IHC images:

```
CUDA_VISIBLE_DEVICES=0,1 python inference.py --stain p53 \\
--network convnext_large \\
--model_path <path to p53 model (p53 convnext_large)> \\
--slide_path <path to raw slide files to read from> \\
--patch_path <path to hdf5 (.h5) patches from patching.py> \\
--save_dir <path to output folder> \\
--json \\
--process_list <txt or csv files of slides to infer>   
```
</details>

## Collating results
Once all models have been run, results can be collated into one csv file for upload to the RedCap database as part of the ongoing BEST4 trial:
```
python crf.py
```

***
If you use this code please cite the original paper using the citation below.

Gastric cardia detection and TFF3 counting:
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

Atypia and P53 detection:
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
