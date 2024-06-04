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
import torch.nn as nn

from dataset_processing.masked_dataset import MaskedPatchWSIDataset
import monai.transforms as mt
from monai.data import DataLoader

from dataset_processing import class_parser
from models import get_network

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
	parser.add_argument("--tile_size", default=400, help="tile size to extract from slide", type=int)
	parser.add_argument("--patch_size", default=256, help="architecture tile size", type=int)
	parser.add_argument("--patch_level", default=0, help="level of the slide to extract tiles from", type=int)
	parser.add_argument("--mask_level", default=7, help="Mask level for foreground filtering", type=int)
	parser.add_argument("--mask_path", default='masks', help="path to mask for foreground filtering")
	parser.add_argument("--overlap", default=0, help="what fraction of the tile edge neighboring tiles should overlap horizontally and vertically (default is 0)")
	parser.add_argument("--reader", default='openslide', help="monai slide backend reader ('openslide' or 'cuCIM')")

	#data processing
	parser.add_argument("--channel_means", default=[0.7747305964175918, 0.7421753839460998, 0.7307385516144509], help="0-1 normalized color channel means for all tiles on dataset separated by commas, e.g. 0.485,0.456,0.406 for RGB, respectively. Otherwise, provide a path to a 'channel_means_and_stds.pickle' file")
	parser.add_argument("--channel_stds", default=[0.2105364799974944, 0.2123423033814637, 0.20617556948731974], help="0-1 normalized color channel standard deviations for all tiles on dataset separated by commas, e.g. 0.229,0.224,0.225 for RGB, respectively. Otherwise, provide a path to a 'channel_means_and_stds.pickle' file")
	parser.add_argument("--batch_size", default=None, help="Batch size. Default is to use values set for architecture to run on 1 GPU.")
	parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to call for DataLoader")
	
	parser.add_argument("--he_threshold", default=0.99993, type=float, help="A threshold for detecting gastric cardia in H&E")
	parser.add_argument("--p53_threshold", default= 0.99, type=float, help="A threshold for Goblet cell detection in p53")
	parser.add_argument("--lcp_cutoff", default=None, help='number of tiles to be considered low confidence positive')
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

def torchmodify(name):
	a = name.split('.')
	for i,s in enumerate(a) :
		if s.isnumeric() :
			a[i]="_modules['"+s+"']"
	return '.'.join(a)

if __name__ == '__main__':
	args = parse_args()
	
	slide_path = args.slide_path
	patch_level = args.patch_level
	tile_size = args.tile_size
	mask_level = args.mask_level
	network = args.network
	stain = args.stain
	output_path = args.output
	reader = args.reader

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
	
	# Modify the model to use the updated GELU activation function in later PyTorch versions 
	for name, module in trained_model.named_modules() :
		if isinstance(module, nn.GELU):
			exec('trained_model.'+torchmodify(name)+'=nn.GELU()')

	# if args.multi_gpu:
		# trained_model = torch.nn.parallel.DistributedDataParallel(trained_model, device_ids=[args.local_rank], output_device=args.local_rank)

	# Use manual batch size if one has been specified
	if args.batch_size is not None:
		batch_size = args.batch_size
	else:
		batch_size = params['batch_size']
	patch_size = params['patch_size']

	if torch.cuda.is_available() and (torch.version.hip or torch.version.cuda):
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	trained_model.to(device)
	trained_model.eval()

	if not os.path.exists(output_path):
		os.makedirs(output_path, exist_ok=True)
	print("Outputting inference to: ", output_path)

	csv_path = os.path.join(output_path, args.description + '-prediction-data.csv')

	channel_means = args.channel_means
	channel_stds = args.channel_stds
	if not args.silent:
		print('Channel Means: ', channel_means, '\nChannel Stds: ', channel_stds)

	post_process_transforms = mt.Compose(
			[
				mt.ToMetaTensord(keys=("image")),
				mt.Resized(keys="image", spatial_size=(patch_size,patch_size)),
				mt.NormalizeIntensityd(keys="image", subtrahend=channel_means, 
									divisor=channel_stds, channel_wise=True),
			]
		)

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
		if os.path.isfile(slide_output + '.csv'):
			print(f'Inference for {slide_name} already exists.')
			tiles = pd.read_csv(slide_output+'.csv')
		else:
			slide = [{"image": f"{case_path}"}]
			dataset = MaskedPatchWSIDataset(slide, patch_size=patch_size, patch_level=patch_level, mask_level=mask_level, transform=post_process_transforms, reader=reader, additional_meta_keys=["location", "name"])

			since = time.time()
			dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
			tile_predictions = []

			with torch.no_grad():
				print(f'\rCase {case_number}/{len(labels)} {index} processing: ')
				for inputs in tqdm(dataloader, disable=args.silent):
					tile = inputs['image'].to(device)
					output = trained_model(tile)
					output = output.to(device)

					batch_prediction = torch.nn.functional.softmax(output, dim=1).cpu().data.numpy()
					for index in range(len(tile)):
						preds = batch_prediction[index, ...].tolist()
						if len(preds) != len(classes):
							raise ValueError('Model has '+str(len(preds))+' classes but '+str(len(classes))+' class names were provided in the classes argument')
						tile_location = inputs['image'].meta['location'][index].numpy()
						prediction = {'x': tile_location[0], 'y': tile_location[1]}
						for i, pred in enumerate(preds):
							prediction[classes[i]] = pred
						tile_predictions.append(prediction)

			tiles = pd.DataFrame(tile_predictions)
			if args.csv:
				tiles.to_csv(slide_output+'.csv', index=False)

			time_elapsed = time.time() - since
			if not args.silent:
				print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

		counts = (tiles[ranked_class] > thresh).sum()

		data = {'CYT ID': index, 'Case Path': case_file, str(ranked_class+' Tiles'): counts}
		data_list.append(data)
	
		if args.xml or args.json:
			positive = tiles[tiles[ranked_class] > thresh]

			annotation_path = os.path.join(output_path, args.description + '_tile_annotations')
			os.makedirs(annotation_path, exist_ok=True)
			
			if len(positive) == 0:
				print(f'No annotations found at current threshold for {slide_name}')
			else:
				if args.xml:
					xml_path = os.path.join(annotation_path, slide_name+'_inference.xml')
					if not os.path.exists(xml_path):
						# Make ASAP file
						xml_header = """<?xml version="1.0"?><ASAP_Annotations>\t<Annotations>\n"""
						xml_tail = 	f"""\t</Annotations>\t<AnnotationGroups>\t\t<Group Name="{ranked_class}" PartOfGroup="None" Color="#64FE2E">\t\t\t<Attributes />\t\t</Group>\t</AnnotationGroups></ASAP_Annotations>\n"""

						xml_annotations = ""
						for index, row in positive.iterrows():
							xml_annotations = (xml_annotations +
												"\t\t<Annotation Name=\""+str(row[ranked_class+'_probability'])+"\" Type=\"Polygon\" PartOfGroup=\""+ranked_class+"\" Color=\"#F4FA58\">\n" +
												"\t\t\t<Coordinates>\n" +
												"\t\t\t\t<Coordinate Order=\"0\" X=\""+str(row['x'])+"\" Y=\""+str(row['y'])+"\" />\n" +
												"\t\t\t\t<Coordinate Order=\"1\" X=\""+str(row['x']+tile_size)+"\" Y=\""+str(row['y'])+"\" />\n" +
												"\t\t\t\t<Coordinate Order=\"2\" X=\""+str(row['x']+tile_size)+"\" Y=\""+str(row['y']+tile_size)+"\" />\n" +
												"\t\t\t\t<Coordinate Order=\"3\" X=\""+str(row['x'])+"\" Y=\""+str(row['y']+tile_size)+"\" />\n" +
												"\t\t\t</Coordinates>\n" +
												"\t\t</Annotation>\n")
						print('Creating automated annotation file for '+ slide_name)
						with open(xml_path, "w") as annotation_file:
							annotation_file.write(xml_header + xml_annotations + xml_tail)
					else:
						print(f'Automated xml annotation file for {index} already exists.')

				if args.json:
					json_path = os.path.join(annotation_path, slide_name+'_inference.geojson')
					if not os.path.exists(json_path):
						json_annotations = {"type": "FeatureCollection", "features":[]}
						for index, row in positive.iterrows():
							color = [0, 0, 255]
							status = str(ranked_class)

							json_annotations['features'].append({
								"type": "Feature",
								"id": "PathDetectionObject",
								"geometry": {
								"type": "Polygon",
								"coordinates": [
										[
											[row['x'], row['y']],
											[row['x']+tile_size, row['y']],
											[row['x']+tile_size, row['y']+tile_size],
											[row['x'], row['y']+tile_size],		
											[row['x'], row['y']]
										]	
									]
								},
								"properties": {
									"objectType": "annotation",
									"name": str(status)+'_'+str(round(row[ranked_class], 4))+'_'+str(row['x']) +'_'+str(row['y']),
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
	
	if args.csv:
		df = pd.DataFrame.from_dict(data_list)
		df.to_csv(csv_path, index=False)

	if args.stats:
		df = pd.DataFrame.from_dict(data_list)

		gt_col = df['Ground Truth Label']
		gt = gt_col.tolist()

		pred_col = df[ranked_class]
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
