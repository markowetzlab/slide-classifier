# This file runs tile-level inference on the training, calibration and internal validation cohort

import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from slidl.slide import Slide

from dataset_processing.image import image_transforms
from models import get_network
from utils.visualisation.display_slide import slide_image
from slidl.utils.torch.WholeSlideImageDataset import WholeSlideImageDataset

from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, classification_report,
			     confusion_matrix)

import warnings
warnings.filterwarnings('always')

def parse_args():
	parser = argparse.ArgumentParser(description='Run inference on slides.')

	#dataset processing
	parser.add_argument("--stain", required=True, help="he or tff3")
	parser.add_argument("--labels", help="file containing slide-level ground truth to use.'")

	#model path and parameters
	parser.add_argument("--network", default='vgg_16', help="which CNN architecture to use")
	parser.add_argument("--model_path", required=True, help="path to stored model weights")

	#slide paths and tile properties
	parser.add_argument("--slide_path", required=True, help="slides root folder")
	parser.add_argument("--format", default=".ndpi", help="extension of whole slide image")
	parser.add_argument("--tile_size", default=400, help="architecture tile size")
	parser.add_argument("--overlap", default=0, help="what fraction of the tile edge neighboring tiles should overlap horizontally and vertically (default is 0)")
	parser.add_argument("--foreground_only", action='store_true', help="Foreground with tissue only")
	
	#data processing
	parser.add_argument("--channel_means", default= [0.485, 0.456, 0.406], help="0-1 normalized color channel means for all tiles on dataset separated by commas, e.g. 0.485,0.456,0.406 for RGB, respectively. Otherwise, provide a path to a 'channel_means_and_stds.pickle' file")
	parser.add_argument("--channel_stds", default= [0.229, 0.224, 0.225], help="0-1 normalized color channel standard deviations for all tiles on dataset separated by commas, e.g. 0.229,0.224,0.225 for RGB, respectively. Otherwise, provide a path to a 'channel_means_and_stds.pickle' file")
	parser.add_argument("--batch_size", default=None, help="Batch size. Default is to use values set for architecture to run on 1 GPU.")
	parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to call for DataLoader")
	
	parser.add_argument("--qc_threshold", default=0.99, help="A threshold for detecting gastric cardia in H&E")
	parser.add_argument("--tff3_threshold", default= 0.93, help="A threshold for Goblet cell detection in tff3")
	parser.add_argument("--tile_cutoff", default=6, help='number of tiles to be considered positive')

	#outputs
	parser.add_argument("--output", required=True, help="path to folder where inference maps will be stored")
	parser.add_argument("--csv", action='store_true', help="Generate csv output file")
	parser.add_argument("--vis", action='store_true', help="Display WSI after each slide")
	parser.add_argument("--thumbnail", action='store_true', help="Save thumbnail of WSI for analysis (vis must also be true)")
	parser.add_argument('--stats', action='store_true', help='produce precision-recall plot')

	parser.add_argument('--silent', action='store_true', help='Flag which silences terminal outputs')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	
	slide_path = args.slide_path
	tile_size = args.tile_size
	network = args.network
	stain = args.stain
	output_path = args.output

	if stain == 'he':
		file_name = 'H&E'
		gt_label = 'QC Report'
		classes = ['Background', 'Gastric-type columnar epithelium', 'Intestinal Metaplasia', 'Respiratory-type columnar epithelium']
		ranked_class = 'Gastric-type columnar epithelium'
		secondary_class = 'Intestinal Metaplasia'
		thresh = args.qc_threshold
		mapping = {'Adequate for pathological review': 1, 'Scant columnar cells': 0, 'Squamous cells only': 0, 'Insufficient cellular material': 0}
	elif stain == 'tff3':
		file_name = 'TFF3'
		gt_label = 'TFF3 positive'
		classes = ['Equivocal', 'Negative', 'Positive']
		ranked_class = 'Positive'
		secondary_class = 'Equivocal'
		thresh = args.tff3_threshold
		mapping = {'Y': 1, 'N': 0}
	else:
		raise AssertionError('Stain must be he or tff3.')

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
		device = torch.device("cuda:0")
	elif torch.cuda.is_available() and torch.version.cuda:
		device = torch.device("cuda:0")
	else:
		device = torch.device("cpu")

	torch.nn.Module.dump_patches = True

	trained_model.to(device)
	trained_model.eval()

	if not os.path.exists(output_path):
		os.makedirs(output_path, exist_ok=True)
	print("Outputting inference to: ", output_path)

	csv_path = os.path.join(output_path, network + '-' + stain + '-prediction-data.csv')

	channel_means = args.channel_means
	channel_stds = args.channel_stds
	if not args.silent:
		print('Channel Means: ', channel_means, '\nChannel Stds: ', channel_stds)

	data_transforms = image_transforms(channel_means, channel_stds, patch_size)['val']

	if os.path.isfile(args.labels):
		labels = pd.read_csv(args.labels, index_col=0)
		labels[gt_label] = labels[gt_label].fillna('N')
		labels.sort_index(inplace=True)
		labels[gt_label] = labels[gt_label].map(mapping)
	else:
		raise AssertionError('Not a valid path for ground truth labels.')

	data_list = []

	case_number = 0
	for index, row in labels.iterrows():
		case_file = row[file_name]
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
			print(f'Processing case {case_number}/{len(labels)}: ', end='')

		inference_output = os.path.join(output_path, 'triage')
		if not os.path.exists(inference_output):
			os.makedirs(inference_output)
		slide_output = os.path.join(inference_output, case_file.replace(args.format, '_triage'))

		if os.path.isfile(slide_output + '.pml'):
			if not args.silent:
				print(f"Case {index} already processed")
			slidl_slide = Slide(slide_output + '.pml', newSlideFilePath=case_path)
		else:
			slidl_slide = Slide(case_path).setTileProperties(tileSize=tile_size, tileOverlap=float(args.overlap))

			if args.foreground_only:
				slidl_slide.detectForeground(threshold=95)

			dataset = WholeSlideImageDataset(slidl_slide, foregroundOnly=args.foreground_only, transform=data_transforms)

			since = time.time()
			dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
			tile_predictions = []
			
			with torch.no_grad():
				for inputs in tqdm(dataloader, disable=args.silent):
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
			class_tiles = slidl_slide.numTilesAboveClassPredictionThreshold(classToThreshold=class_name, probabilityThresholds=thresh)
			data.update({class_name+' Tiles': class_tiles})
		data_list.append(data)

		if args.vis:
			slide_im = slide_image(slidl_slide, stain, classes)
			im_path = os.path.join(args.output, 'images')
			if not os.path.exists(im_path):
				os.makedirs(im_path)
			slide_im.plot_thumbnail(case_id=index, target=ranked_class)
			if args.thumbnail:
				slide_im.save(im_path, case_file.replace(args.format, "_thumbnail"))
			slide_im.draw_class(target = ranked_class)
			slide_im.plot_class(target = ranked_class)
			slide_im.save(im_path, case_file.replace(args.format, "_" + ranked_class))
			# plt.show()
			plt.close('all')

	df = pd.DataFrame.from_dict(data_list)
	if args.csv:
		df.to_csv(csv_path, index_label='CYT ID')

	if args.stats:
		gt_col = df[gt_label]
		gt = gt_col.tolist()

		pred_col = df[ranked_class+ ' Tiles']
		pred = (pred_col > args.tile_cutoff).astype(int).tolist()

		auc_data = roc_auc_score(gt, pred)
		auprc_data = average_precision_score(gt, pred)
		
		fpr, tpr, threshs = roc_curve(gt, pred)
		precision, recall, thresholds = precision_recall_curve(gt_col, pred_col)

		print(confusion_matrix(gt, pred))
		print(classification_report(gt, pred))
