# This file runs tile-level inference on the training, calibration and internal validation cohort

import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geojson

from tqdm import tqdm

import torch
from slidl.slide import Slide

from legacy.image import image_transforms
from legacy.display_slide import slide_image
from slidl.utils.torch.WholeSlideImageDataset import WholeSlideImageDataset

from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
				classification_report, confusion_matrix)

import warnings
warnings.filterwarnings('ignore')

def parse_args():
	parser = argparse.ArgumentParser(description='Run inference on slides.')

	parser.add_argument("--description", default='diagnosis', help="string to save results to.")

	#dataset processing
	parser.add_argument("--stain", choices=['he', 'qc', 'p53'], help="H&E (he), quality control (qc) or P53 (p53)")
	parser.add_argument("--labels", default=None, help="file containing slide-level ground truth to use.'")

	#model path and parameters
	parser.add_argument("--network", default='vgg_16', help="which CNN architecture to use")
	parser.add_argument("--model_path", required=True, help="path to stored model weights")

	#slide paths and tile properties
	parser.add_argument("--slide_path", default='DELTA/slides', help="slides root folder")
	parser.add_argument("--format", default=".ndpi", help="extension of whole slide image")
	parser.add_argument("--tile_size", default=400, help="architecture tile size")
	parser.add_argument("--overlap", default=0, help="what fraction of the tile edge neighboring tiles should overlap horizontally and vertically (default is 0)")
	parser.add_argument("--foreground_only", action='store_true', help="Foreground with tissue only")
	parser.add_argument("--tissue_only", action='store_true', help="Apply TissueTector to segment Tissue only")
	
	#data processing
	parser.add_argument("--channel_means", default=[0.7747305964175918, 0.7421753839460998, 0.7307385516144509], help="0-1 normalized color channel means for all tiles on dataset separated by commas, e.g. 0.485,0.456,0.406 for RGB, respectively. Otherwise, provide a path to a 'channel_means_and_stds.pickle' file")
	parser.add_argument("--channel_stds", default=[0.2105364799974944, 0.2123423033814637, 0.20617556948731974], help="0-1 normalized color channel standard deviations for all tiles on dataset separated by commas, e.g. 0.229,0.224,0.225 for RGB, respectively. Otherwise, provide a path to a 'channel_means_and_stds.pickle' file")
	parser.add_argument("--batch_size", default=None, help="Batch size. Default is to use values set for architecture to run on 1 GPU.")
	parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to call for DataLoader")
	
	parser.add_argument("--he_threshold", default=0.999, type=float, help="A threshold for detecting gastric cardia in H&E")
	parser.add_argument("--p53_threshold", default= 0.99, type=float, help="A threshold for Goblet cell detection in p53")
	parser.add_argument("--lcp_cutoff", default=None, help='number of tiles to be considered high confidence negative')
	parser.add_argument("--hcp_cutoff", default=None, help='number of tiles to be considered high confidence positive')
	parser.add_argument("--impute", action='store_true', help="Assume missing data as negative")
	parser.add_argument("--control", default=None, help='csv containing control tissue location from control.py.')

	#class variables
	parser.add_argument("--dysplasia_separate", action='store_false', help="Flag whether to separate the atypia of uncertain significance and dysplasia classes")
	parser.add_argument("--respiratory_separate", action='store_false', help="Flag whether to separate the respiratory mucosa cilia and respiratory mucosa classes")
	parser.add_argument("--gastric_separate", action='store_false', help="Flag whether to separate the tickled up columnar and gastric cardia classes")
	parser.add_argument("--atypia_separate", action='store_false', help="Flag whether to perform the following class split: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other")
	parser.add_argument("--p53_separate", action='store_false', help="Flag whether to perform the following class split: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar")

	parser.add_argument("--ranked_class", default=None, help='')

	#outputs
	parser.add_argument("--output", default='results', help="path to folder where inference maps will be stored")
	parser.add_argument("--csv", action='store_true', help="Generate csv output file")
	parser.add_argument("--stats", action='store_true', help='produce precision-recall plot')
	parser.add_argument("--xml", action='store_true', help='produce annotation files for ASAP in .xml format')
	parser.add_argument("--json", action='store_true', help='produce annotation files for QuPath in .geoJSON format')
	parser.add_argument("--vis", action='store_true', help="Display WSI with heatmap after each slide")
	parser.add_argument("--thumbnail", action='store_true', help="Save thumbnail of WSI for analysis (vis must also be true)")

	parser.add_argument('--silent', action='store_true', help='Flag which silences terminal outputs')

	args = parser.parse_args()
	return args

def class_parser(stain='he', dysplasia_combine = True, respiratory_combine = True, gastric_combine = True, atypia_combine = True, p53_combine = True):
    '''
    stain - 'he' or 'p53'
    dysplasia_combine - combine the atypia of uncertain significance and dysplasia classes
    respiratory_combine - combine the respiratory mucosa cilia and respiratory mucosa classes
    gastric_combine - combine the tickled up columnar and gastric cardia classes
    atypia_combine - perform the following class mergers: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other
    p53_combine - perform the following class mergers: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar
    '''
    
    class_names = ['artifact', 'background', 'immune_cells', 'squamous_mucosa']
    he_classes = ['gastric_cardia', 'intestinal_metaplasia', 'respiratory_mucosa']
    p53_classes = ['aberrant_positive_columnar', 'wild_type_columnar']

    if stain == "he":
        class_names.extend(he_classes)
        if atypia_combine:
            class_names.extend(['atypia'])
        else:
            if dysplasia_combine:
                class_names.extend(['other'])
                if respiratory_combine:
                    if not gastric_combine:
                        class_names.extend(['tickled_up_columnar', 'atypia'])
                else:
                    class_names.extend('respiratory_mucosa_cilia')
                    if gastric_combine:
                        class_names.extend(['atypia'])
                    else:
                        class_names.extend(['tickled_up_columnar', 'atypia'])

            else:
                class_names.extend([['atypia_of_uncertain_significance', 'other', 'dysplasia']])
                if not respiratory_combine:
                    class_names.extend('respiratory_mucosa_cilia')
                if not gastric_combine:
                        class_names.extend(['tickled_up_columnar'])
    
    else:
        class_names.extend(p53_classes)
        if not p53_combine:
            class_names.extend(['equivocal_columnar', 'nonspecific_background', 'oral_bacteria', 'respiratory_mucosa'])

    class_names.sort()
    return class_names

import torch.nn as nn
from torchvision import models

def get_network(network_name, class_names, pretrained=True): 
    params = {'network_name': str(network_name), 'classes': class_names}

    model = ['vgg11_bn', 'vgg16', 'vgg19_bn', 'resnet18', 'resnet152', 'squeezenet1_1', 'densenet121', 'densenet161', 'alexnet',
                 'resnext101_32x8d', 'googlenet', 'shufflenet_v2_x1_0', 'mobilenet_v3_large', 'wide_resnet101_2', 'mnasnet1_0',
                 'efficientnet_b0', 'efficientnet_b7', 'regnet_x_32gf', 'regnet_y_32gf', 'vit_b_16', 'vit_l_16', 'convnext_tiny', 'convnext_large', 'inception_v3']

    if params['network_name'] not in model:
        print(model)
        raise NotImplementedError('Please choose from the available networks above.')

    if network_name == 'inception_v3': 
        params['patch_size'] = 299
    else: 
        params['patch_size'] = 224

    batch_sizes = {"resnet18": 438, "resnet152": 50, "vgg11_bn": 94, "vgg16": 90, 
    "vgg19_bn": 52, "inception_v3": 96, "alexnet": 1988, "squeezenet1_1": 490, 
    "densenet121": 75, "densenet161": 37, "resnext101_32x8d": 32, "googlenet": 206, 
    "shufflenet_v2_x1_0": 480, "mobilenet_v3_large": 222, "wide_resnet101_2": 46, 
    "mnasnet1_0": 220, "efficientnet_b0": 11, "efficientnet_b7": 1, "regnet_x_32gf": 27, 
    "regnet_y_32gf": 27, "vit_b_16": 51, "vit_l_16": 10, "convnext_tiny": 86, "convnext_large": 17 
    }

    params['batch_size'] = batch_sizes[network_name]

    if network_name == "resnet18":
        model_ft = models.resnet18(pretrained=pretrained)
        model_ft.fc = nn.Linear(512, len(class_names))
    elif network_name == "resnet152":
        model_ft = models.resnet152(pretrained=pretrained)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
    elif network_name == "inception_v3":
        model_ft = models.inception_v3(pretrained=pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(768, len(class_names))
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    elif network_name == "vgg11_bn":
        model_ft = models.vgg11_bn(pretrained=pretrained)
        model_ft.classifier[6] = nn.Linear(4096, len(class_names))
    elif network_name == "vgg16":
        model_ft = models.vgg16(pretrained=pretrained)
        model_ft.classifier[6] = nn.Linear(4096, len(class_names))
    elif network_name == "vgg19_bn":
        model_ft = models.vgg19_bn(pretrained=pretrained)
        model_ft.classifier[6] = nn.Linear(4096, len(class_names))
    elif network_name == "densenet121":
        model_ft = models.densenet121(pretrained=pretrained)
        model_ft.classifier = nn.Linear(1024, len(class_names))
    elif network_name == "densenet161":
        model_ft = models.densenet161(pretrained=pretrained)
        model_ft.classifier = nn.Linear(2208, len(class_names))
    elif network_name == "alexnet":
        model_ft = models.alexnet(pretrained=pretrained)
        model_ft.classifier[6] = nn.Linear(4096, len(class_names))
    elif network_name == "squeezenet1_1":
        model_ft = models.squeezenet1_1(pretrained=pretrained)
        model_ft.classifier[1] = nn.Conv2d(
            512, len(class_names), kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = len(class_names)
    elif network_name == "resnext101_32x8d":
        model_ft = models.resnext101_32x8d(pretrained=pretrained)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
    elif network_name == "googlenet":
        model_ft = models.googlenet(pretrained=pretrained)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
    elif network_name == "shufflenet_v2_x1_0":
        model_ft = models.shufflenet_v2_x1_0(pretrained=pretrained)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
    elif network_name == "mobilenet_v3_large":
        model_ft = models.mobilenet_v3_large(pretrained=pretrained)
        model_ft.classifier[-1] = nn.Linear(1280, len(class_names))
    elif network_name == "wide_resnet101_2":
        model_ft = models.wide_resnet101_2(pretrained=pretrained)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
    elif network_name == "mnasnet1_0":
        model_ft = models.mnasnet1_0(pretrained=pretrained)
        model_ft.classifier[-1] = nn.Linear(1280, len(class_names))
    elif network_name == "efficientnet_b0":
        model_ft = models.efficientnet_b0(pretrained=pretrained)
        model_ft.classifier[-1] = nn.Linear(1280, len(class_names))
    elif network_name == "efficientnet_b7":
        model_ft = models.efficientnet_b7(pretrained=pretrained)
        model_ft.classifier[-1] = nn.Linear(2560, len(class_names))
    elif network_name == "regnet_x_32gf":
        model_ft = models.regnet_x_32gf(pretrained=pretrained)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
    elif network_name == "regnet_y_32gf":
        model_ft = models.regnet_y_32gf(pretrained=pretrained)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
    elif network_name == "vit_b_16":
        model_ft = models.vit_b_16(pretrained=pretrained)
        model_ft.heads[-1] = nn.Linear(768, len(class_names))
    elif network_name == "vit_l_16":
        model_ft = models.vit_l_16(pretrained=pretrained)
        model_ft.heads[-1] = nn.Linear(1024, len(class_names))
    elif network_name == "convnext_tiny":
        model_ft = models.convnext_tiny(pretrained=pretrained)
        model_ft.classifier[-1] = nn.Linear(768, len(class_names))
    elif network_name == "convnext_large":
        model_ft = models.convnext_large(pretrained=pretrained)
        model_ft.classifier[-1] = nn.Linear(1536, len(class_names))

    return model_ft, params

if __name__ == '__main__':
	args = parse_args()
	
	slide_path = args.slide_path
	tile_size = args.tile_size
	network = args.network
	stain = args.stain
	output_path = args.output

	if stain == 'he':
		file_name = 'H&E'
		gt_label = 'Atypia'
		classes = class_parser('he', args.dysplasia_separate, args.respiratory_separate, args.gastric_separate, args.atypia_separate, args.p53_separate)
		if args.ranked_class is not None:
			ranked_class = args.ranked_class
		else:
			ranked_class = 'atypia'
		thresh = args.he_threshold
		mapping = {'Y': 1, 'N': 0}
		if args.lcp_cutoff is not None:
			lcp_threshold = args.lcp_cutoff
		else:
			lcp_threshold = 0 
		if args.hcp_cutoff is not None:
			hcp_threshold = args.hcp_cutoff
		else:
			hcp_threshold = 10
	elif stain == 'qc':
		file_name = 'H&E'
		gt_label = 'QC Report'
		classes = class_parser('he', args.dysplasia_separate, args.respiratory_separate, args.gastric_separate, args.atypia_separate, args.p53_separate)
		mapping = {'Adequate for pathological review': 1, 'Scant columnar cells': 1, 'Squamous cells only': 0, 'Insufficient cellular material': 0, 'Food material': 0}
		if args.ranked_class is not None:
			ranked_class = args.ranked_class
		else:
			ranked_class = 'gastric_cardia'
		thresh = args.he_threshold
		if args.lcp_cutoff is not None:
			lcp_threshold = args.lcp_cutoff
		else:
			lcp_threshold = 0
		if args.hcp_cutoff is not None:
			hcp_threshold = args.hcp_cutoff
		else:
			hcp_threshold = 95
	elif stain == 'p53':
		file_name = 'P53'
		gt_label = 'P53 positive'
		classes = class_parser('p53', args.p53_separate)
		if args.ranked_class is not None:
			ranked_class = args.ranked_class
		else:
			ranked_class = 'aberrant_positive_columnar'
		thresh = args.p53_threshold
		mapping = {'Y': 1, 'N': 0}
		if args.lcp_cutoff is not None:
			lcp_threshold = args.lcp_cutoff
		else:
			lcp_threshold = 0
		if args.hcp_cutoff is not None:
			hcp_threshold = args.hcp_cutoff
		else:
			hcp_threshold = 2
	else:
		raise AssertionError('Stain must be he/qc or p53.')

	trained_model, params = get_network(network, class_names=classes, pretrained=False)
	try:
		trained_model.load_state_dict(torch.load(args.model_path).module.state_dict())
	except: 
		trained_model = torch.load(args.model_path)
	
	# Use manual batch size if one has been specified
	if args.batch_size is not None:
		batch_size = args.batch_size
	else:
		batch_size = params['batch_size']
	patch_size = params['patch_size']

	if torch.cuda.is_available() and torch.version.hip:
		device = torch.device("cuda")
	elif torch.cuda.is_available() and torch.version.cuda:
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	trained_model.to(device)
	trained_model.eval()

	if not os.path.exists(output_path):
		os.makedirs(output_path, exist_ok=True)
	print("Outputting inference to: ", output_path)

	csv_path = os.path.join(output_path, args.description + '-prediction-data.csv')
	if args.control is not None:
		controls = pd.read_csv(args.control, index_col='CYT ID')

	channel_means = args.channel_means
	channel_stds = args.channel_stds
	if not args.silent:
		print('Channel Means: ', channel_means, '\nChannel Stds: ', channel_stds)

	data_transforms = image_transforms(channel_means, channel_stds, patch_size)['val']

	if args.labels is not None:
		if os.path.isfile(args.labels):
			labels = pd.read_csv(args.labels, index_col=0)
			labels.dropna(subset=[file_name], inplace=True)
			if args.impute:
				labels[gt_label] = labels[gt_label].fillna('N')
			else:
				labels.dropna(subset=[gt_label], inplace=True)
			labels.sort_index(inplace=True)
			labels[gt_label] = labels[gt_label].map(mapping)
		else:
			raise AssertionError('Not a valid path for ground truth labels.')
	else:
		slides = []
		for file in os.listdir(slide_path):
			if file.endswith(('.ndpi','.svs')):
				slides.append(file)
		slides = sorted(slides)
		labels = pd.DataFrame(slides, columns=[file_name])
		labels[gt_label] = 0

	data_list = []

	case_number = 0
	for index, row in labels.iterrows():
		case_file = row[file_name]
		slide_name = case_file.replace(args.format, "")
		case_number += 1
		try:
			case_path = os.path.join(slide_path, case_file)
		except:
			print(f'File {file_name} not found.')
			continue

		if not os.path.exists(case_path):
			print(f'File {case_path} not found.')
			continue
		if not args.silent:
			print(f'\rProcessing case {case_number}/{len(labels)}: ', end='')

		inference_output = os.path.join(output_path, 'inference')
		if not os.path.exists(inference_output):
			os.makedirs(inference_output)
		slide_output = os.path.join(inference_output, slide_name+'_inference')

		if os.path.isfile(slide_output + '.pml'):
			if not args.silent:
				print(f'\rCase {case_number}/{len(labels)} {index} already processed.', end='\r')
			slidl_slide = Slide(slide_output + '.pml', newSlideFilePath=case_path)
		else:
			slidl_slide = Slide(case_path).setTileProperties(tileSize=tile_size, tileOverlap=float(args.overlap))

			if args.foreground_only:
				slidl_slide.detectForeground(threshold=95)
			if args.tissue_only:
				slidl_slide.TissueTect()

			dataset = WholeSlideImageDataset(slidl_slide, foregroundOnly=args.foreground_only, transform=data_transforms)

			since = time.time()
			dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
			tile_predictions = []
			
			with torch.no_grad():
				print(f'\rCase {case_number}/{len(labels)} {index} processing: ')
				for inputs in tqdm(dataloader, disable=args.silent):
					import ipdb; ipdb.set_trace()
					inputTile = inputs['image'].to(device)
					output = trained_model(inputTile)
					output = output.to(device)

					batch_prediction = torch.nn.functional.softmax(output, dim=1).cpu().data.numpy()

					for index in range(len(inputTile)):
						tileAddress = (inputs['tileAddress'][0][index].item(), inputs['tileAddress'][1][index].item())
						preds = batch_prediction[index, ...].tolist()
						if len(preds) != len(classes):
							raise ValueError('Model has '+str(len(preds))+' classes but '+str(len(classes))+' class names were provided in the classes argument')
						prediction = {}
						for i, pred in enumerate(preds):
							prediction[classes[i]] = pred
						slidl_slide.appendTag(tileAddress, 'classifierInferencePrediction', prediction)
						tile_predictions.append(tileAddress)			
			
			slidl_slide.save(fileName = slide_output)
			time_elapsed = time.time() - since
			if not args.silent:
				print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

		data = {'CYT ID': index, 'Case Path': case_file, 'Ground Truth Label': row[gt_label]}

		for class_name in classes:
			if args.control is not None:
				X_tiles = slidl_slide.numTilesInX
				Y_tiles = slidl_slide.numTilesInY
				X_control = round(X_tiles/4)
				Y_control = round(Y_tiles/4)
	
				control_loc = controls.loc[index]
				class_tiles = 0
				for tile_address, tile_entry in slidl_slide.tileDictionary.items():
					if tile_entry['classifierInferencePrediction'][class_name] >= thresh:
						if tile_address[0] < X_control and control_loc['L'] == 1:
							continue
						elif X_tiles - X_control < tile_address[0] and control_loc['R'] == 1:
							continue
						elif Y_tiles - Y_control < tile_address[1] and (control_loc['B'] == 1 and (control_loc['L'] == 0 or control_loc['R'] == 0)):
							continue
						elif tile_address[1] < Y_control and (control_loc['T'] == 1 and (control_loc['L'] == 0 or control_loc['R'] == 0)):
							continue
						else:
							class_tiles += 1
			else:
				class_tiles = slidl_slide.numTilesAboveClassPredictionThreshold(classToThreshold=class_name, probabilityThresholds=float(thresh))
			data.update({class_name+' Tiles': class_tiles})
		data_list.append(data)

		if args.xml or args.json:
			ranked_dict = {}
			annotation_path = os.path.join(output_path, args.description + '_tile_annotations')
			os.makedirs(annotation_path, exist_ok=True)

			for tile_address, tile_entry in slidl_slide.tileDictionary.items():
				for class_name, prob in tile_entry['classifierInferencePrediction'].items():
					if class_name == ranked_class:
						if prob >= float(thresh):
							ranked_dict[str(class_name)+'_'+str(tile_entry['x'])+'_'+str(tile_entry['y'])] = {class_name+'_probability': prob, 'x': tile_entry['x'], 'y': tile_entry['y'], 'width': tile_entry['width']}

			if len(ranked_dict) == 0:
				print(f'No annotations found at current threshold for {slide_name}')
			else:
				if args.xml:
					xml_path = os.path.join(annotation_path, slide_name+'_inference.xml')
					if not os.path.exists(xml_path):
						# Make ASAP file
						xml_header = """<?xml version="1.0"?><ASAP_Annotations>\t<Annotations>\n"""
						xml_tail = 	f"""\t</Annotations>\t<AnnotationGroups>\t\t<Group Name="{ranked_class}" PartOfGroup="None" Color="#64FE2E">\t\t\t<Attributes />\t\t</Group>\t</AnnotationGroups></ASAP_Annotations>\n"""

						xml_annotations = ""
						for key, tile_info in sorted(ranked_dict.items(), reverse=True):
							xml_annotations = (xml_annotations +
												"\t\t<Annotation Name=\""+str(tile_info[ranked_class+'_probability'])+"\" Type=\"Polygon\" PartOfGroup=\""+ranked_class+"\" Color=\"#F4FA58\">\n" +
												"\t\t\t<Coordinates>\n" +
												"\t\t\t\t<Coordinate Order=\"0\" X=\""+str(tile_info['x'])+"\" Y=\""+str(tile_info['y'])+"\" />\n" +
												"\t\t\t\t<Coordinate Order=\"1\" X=\""+str(tile_info['x']+tile_info['width'])+"\" Y=\""+str(tile_info['y'])+"\" />\n" +
												"\t\t\t\t<Coordinate Order=\"2\" X=\""+str(tile_info['x']+tile_info['width'])+"\" Y=\""+str(tile_info['y']+tile_info['width'])+"\" />\n" +
												"\t\t\t\t<Coordinate Order=\"3\" X=\""+str(tile_info['x'])+"\" Y=\""+str(tile_info['y']+tile_info['width'])+"\" />\n" +
												"\t\t\t</Coordinates>\n" +
												"\t\t</Annotation>\n")
						print('Creating automated annotation file for '+slide_name)
						with open(xml_path, "w") as annotation_file:
							annotation_file.write(xml_header + xml_annotations + xml_tail)
					else:
						print(f'Automated xml annotation file for {index} already exists.')

				if args.json:
					json_path = os.path.join(annotation_path, slide_name+'_inference.geojson')
					if not os.path.exists(json_path):
						json_annotations = {"type": "FeatureCollection", "features":[]}
						for key, tile_info in sorted(ranked_dict.items(), reverse=True):
							if ranked_class in key:
								color = [0, 0, 255]
								status = str(ranked_class)
							else:
								color = [0, 0, 0]
								status = str(key)
							
							json_annotations['features'].append({
								"type": "Feature",
								"id": "PathDetectionObject",
								"geometry": {
								"type": "Polygon",
								"coordinates": [
										[
											[tile_info['x'], tile_info['y']],
											[tile_info['x']+tile_info['width'], tile_info['y']],
											[tile_info['x']+tile_info['width'], tile_info['y']+tile_info['width']],
											[tile_info['x'], tile_info['y']+tile_info['width']],		
											[tile_info['x'], tile_info['y']]
										]	
									]
								},
								"properties": {
									"objectType": "annotation",
									"name": str(status)+'_'+str(round(tile_info[status+'_probability'], 4))+'_'+str(tile_info['x']) +'_'+str(tile_info['y']),
									"classification": {
										"name": status,
										"color": color
									}
								}
							})
						print('Creating automated annotation file for ' + slide_name)
						with open(json_path, "w") as annotation_file:
							geojson.dump(json_annotations, annotation_file, indent=0)
					else:
						print(f'Automated geojson annotation file for {index} already exists')

		if args.vis:
			slide_im = slide_image(slidl_slide, stain, classes)
			im_path = os.path.join(args.output, 'images')
			if not os.path.exists(im_path):
				os.makedirs(im_path)
			slide_im.plot_thumbnail(case_id=index, target=ranked_class)
			if args.thumbnail:
				slide_im.save(im_path, slide_name+"_thumbnail")
			slide_im.draw_class(target = ranked_class)
			slide_im.plot_class(target = ranked_class)
			slide_im.save(im_path, slide_name+"_"+ ranked_class)
			plt.show()
			plt.close('all')

	df = pd.DataFrame.from_dict(data_list)

	if args.stats:
		gt_col = df['Ground Truth Label']
		gt = gt_col.tolist()

		pred_col = df[ranked_class+ ' Tiles']
		print(f'\nLCP Tile Threshold ({lcp_threshold})')
		df['LCP'] = pred_col.gt(int(lcp_threshold)).astype(int)
		pred = df[f'LCP'].tolist()

		if args.labels is not None:
			tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
			auc = roc_auc_score(gt, pred)
			precision = precision_score(gt, pred)
			recall = recall_score(gt, pred)
			f1 = f1_score(gt, pred)
			
			print('AUC: ', auc)
			print('Sensitivity: ', recall)
			print('Specificity: ', tn/(tn+fp))
			print('Precision: ', precision)
			print('F1: ', f1, '\n')

			print(f'CM \tGT Positive\tGT Negative')
			print(f'Pred Positive\t{tp}\t{fp}')
			print(f'Pred Negative\t{fn}\t{tn}')
			print(classification_report(gt, pred))
		else:
			print(df['LCP'].value_counts())

		print(f'\nHCP Tile Threshold ({hcp_threshold})')
		df['HCP'] = pred_col.gt(int(hcp_threshold)).astype(int)
		pred = df[f'HCP'].tolist()

		if args.labels is not None:
			tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
			auc = roc_auc_score(gt, pred)
			precision = precision_score(gt, pred)
			recall = recall_score(gt, pred)
			f1 = f1_score(gt, pred)
			
			print('AUC: ', auc)
			print('Sensitivity: ', recall)
			print('Specificity: ', tn/(tn+fp))
			print('Precision: ', precision)
			print('F1: ', f1, '\n')

			print(f'CM \tGT Positive\tGT Negative')
			print(f'Pred Positive\t{tp}\t{fp}')
			print(f'Pred Negative\t{fn}\t{tn}')

			print(classification_report(gt, pred))
		else:
			print(df['HCP'].value_counts())

		results = [
			(df['LCP'] == 0) & (df['HCP'] == 0),
			(df['LCP'] == 1) & (df['HCP'] == 0),
			(df['LCP'] == 1) & (df['HCP'] == 1)
		]
		values = ['High Confidence Negative', 'Low Confidence Positive', 'High Confidence Positive']
		df['Result'] = np.select(results, values)
		print('Predictions: \n', df['Result'].value_counts())

	if args.csv:
		df.to_csv(csv_path, index=False)
