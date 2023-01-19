# Marcel Gehrung and Adam Berman
# This file runs tile-level inference on the training, calibration and internal validation cohort

import argparse
import glob
import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from slidl.slide import Slide

from dataset_processing import class_parser
from dataset_processing.image import channel_averages, image_transforms
from models import get_network
from utils.visualisation.display_slide import slide_image
from utils.metrics.threshold import precision_recall_plots, auc_thresh_plot, roc_thresh_plot, auprc_curve_plot, auprc_thresh_plot
from slidl.utils.torch.WholeSlideImageDataset import WholeSlideImageDataset

from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
							 f1_score, precision_recall_curve, precision_score,
							 recall_score, roc_auc_score, roc_curve)

import warnings
warnings.filterwarnings('always')

def parse_args():
	parser = argparse.ArgumentParser(description='Run inference on slides.')

	#dataset processing
	parser.add_argument("--dataset", default='delta', help="Flag to switch between datasets. Currently supported: 'best'/'delta'")
	parser.add_argument("--stain", required=True, help="he or p53")
	parser.add_argument("--labels", help="file containing slide-level ground truth to use.'")
	parser.add_argument("--target", type=str, default=None, help="Target class to identify, if None then defaults to stain class.")
	parser.add_argument("--gt", type=str, default=None, help="Column containing ground truth labels")

	#model path and parameters
	parser.add_argument("--network", required=True, help="which CNN architecture to use")
	parser.add_argument("--model_path", required=True, help="path to stored model weights")

	#slide paths and tile properties
	parser.add_argument("--slide_path", required=True, help="slides root folder")
	parser.add_argument("--format", default=".ndpi", help="extension of whole slide image")
	parser.add_argument("--tile_size", default=400, help="architecture tile size")
	parser.add_argument("--overlap", default=0, help="what fraction of the tile edge neighboring tiles should overlap horizontally and vertically (default is 0)")
	parser.add_argument("--foreground_only", action='store_true', help="Foreground with tissue only")
	
	#class variables
	parser.add_argument("--dysplasia_separate", action='store_false', help="Flag whether to separate the atypia of uncertain significance and dysplasia classes")
	parser.add_argument("--respiratory_separate", action='store_false', help="Flag whether to separate the respiratory mucosa cilia and respiratory mucosa classes")
	parser.add_argument("--gastric_separate", action='store_false', help="Flag whether to separate the tickled up columnar and gastric cardia classes")
	parser.add_argument("--atypia_separate", action='store_false', help="Flag whether to perform the following class split: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other")
	parser.add_argument("--p53_separate", action='store_false', help="Flag whether to perform the following class split: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar")

	#data processing
	parser.add_argument("--channel_means", default=None, help="0-1 normalized color channel means for all tiles on dataset separated by commas, e.g. 0.485,0.456,0.406 for RGB, respectively. Otherwise, provide a path to a 'channel_means_and_stds.pickle' file")
	parser.add_argument("--channel_stds", default=None, help="0-1 normalized color channel standard deviations for all tiles on dataset separated by commas, e.g. 0.229,0.224,0.225 for RGB, respectively. Otherwise, provide a path to a 'channel_means_and_stds.pickle' file")
	parser.add_argument("--batch_size", default=None, help="Batch size. Default is to use values set for architecture to run on 1 GPU.")
	parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to call for DataLoader")
	
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

	classes = class_parser(args.stain, args.dysplasia_separate, args.respiratory_separate, args.gastric_separate, args.atypia_separate, args.p53_separate)
	trained_model, params = get_network(network, class_names=classes, pretrained=False)
	try:
		trained_model.load_state_dict(torch.load(args.model_path))
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

	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		trained_model = nn.DataParallel(trained_model)

	trained_model.to(device)
	trained_model.eval()

	prob_thresh = np.append(np.arange(0, 0.99, 0.05), np.arange(0.99, 0.999, 0.001))

	output_path = args.output
	if not os.path.exists(output_path):
		os.makedirs(output_path, exist_ok=True)
	print("Outputting inference to: ", output_path)

	if args.target is not None:
		ranked_class = args.target
	elif args.stain == 'he':
		ranked_class = 'atypia'
		if args.gt is not None:
			ranked_label = args.gt
		else:
			ranked_label = 'Atypia'
			file_name = 'H&E'
	elif args.stain == 'p53':
		ranked_class = 'aberrant_positive_columnar'
		if args.gt is not None:
			ranked_label = args.gt
		else:
			ranked_label = 'P53 positive'
			file_name = 'P53'
	else:
		raise AssertionError('Stain currently must be he or p53.')

	csv_path = os.path.join(output_path, args.stain + '-prediction-data.csv')
	if os.path.isfile(csv_path):
		df = pd.read_csv(csv_path, index_col='CYT ID')
	else:
		df = pd.DataFrame()

	channel_means, channel_stds = channel_averages(slide_path, args.channel_means, args.channel_stds)
	if not args.silent:
		print('Channel Means: ', channel_means, '\nChannel Stds: ', channel_stds)
	data_transforms = image_transforms(channel_means, channel_stds, patch_size)['val']

	if os.path.isfile(args.labels):
		labels = pd.read_csv(args.labels, index_col=0)
		labels.sort_index(inplace=True)
		if not args.dataset == 'best':
			labels[ranked_label] = labels[ranked_label].map(dict(Y=1, N=0))
	else:
		raise AssertionError('Not a valid path for ground truth labels.')

	for index, row in labels.iterrows():
		ID = index
		case_file = row[file_name]
		case_path = os.path.join(slide_path, case_file)
		if not os.path.exists(case_path):
			print(f'File {case_path} not found.')
			continue
		if not args.silent:
			print('Processing case: ', ID, 'from', case_file)

		inference_output = os.path.join(output_path, 'inference')
		if not os.path.exists(inference_output):
			os.makedirs(inference_output)
		slide_output = os.path.join(inference_output, case_path.replace(args.format, '_inference'))

		if os.path.isfile(slide_output + '.pml'):
			if not args.silent:
				print(f"Case already processed: {ID}")
			with open(slide_output + '.pml', 'rb') as f:
				tile_dict = pickle.load(f)
			slidl_slide = Slide(slide_output + '.pml')
		else:
			slidl_slide = Slide(case_path).setTileProperties(tileSize=tile_size, tileOverlap=float(args.overlap))

			if args.foreground_only:
				slidl_slide.detectForeground(threshold=95)

			dataset = WholeSlideImageDataset(slidl_slide, foregroundOnly=args.foreground_only, transform=data_transforms)

			since = time.time()
			dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
			
			slidl_slide.inferClassifier(trained_model, classNames=classes, dataTransforms=data_transforms, numWorkers=args.num_workers, silent=args.silent)
			slidl_slide.save(fileName = slide_output)
			time_elapsed = time.time() - since
			if not args.silent:
				print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

		target_accuracy = slidl_slide.numTilesAboveClassPredictionThreshold(ranked_class, probabilityThresholds=[round(prob, 3) for prob in prob_thresh])

		data = {'CYT ID': ID, 'Case Path': case_file, 'Ground Truth Label': int(row[ranked_label])}

		tile_cols = [ranked_class + ' > ' + str(round(prob, 3)) for prob in prob_thresh]
		tile_data = dict(zip(tile_cols, target_accuracy))
		data.update(tile_data)

		df = pd.concat([df, pd.DataFrame([data])])

		if args.vis:
			slide_im = slide_image(slidl_slide, args.stain, classes)
			im_path = os.path.join(args.output, 'images')
			if not os.path.exists(im_path):
				os.makedirs(im_path)
			slide_im.plot_thumbnail(case_id=ID, target=ranked_class)
			if args.thumbnail:
				slide_im.save(im_path, case_path.replace(args.format, "_thumbnail"))
			slide_im.draw_class(target = ranked_class)
			slide_im.plot_class(target = ranked_class)
			plt.show()
			slide_im.save(im_path, case_path.replace(args.format, "_" + ranked_class))

	if args.csv:
		df.to_csv(csv_path)

	if args.stats:
		cutoff_prob = []
		auprc_cutoff_prob = []
	
		cutoffs = {}
		auc_probs = {}
		auc_plotting = {}
		auprc_cutoffs = {}
		auprc_probs = {}
		auprc_plotting = {}
	
		auc_data = []
		auprc_data = []
		fpr_data = []
		tpr_data = []
		precision_data = []
		recall_data = []
	
		binary_recall = []
		binary_precision = []
		binary_f1 = []
	
		for thresh in tile_cols:
			auc_data.append(roc_auc_score(df['Ground Truth Label'], df[thresh]))
			auprc_data.append(average_precision_score(df['Ground Truth Label'], df[thresh]))
			
			fpr, tpr, threshs = roc_curve(df['Ground Truth Label'], df[thresh])
			precision, recall, thresholds = precision_recall_curve(df['Ground Truth Label'], df[thresh])
	
			fpr_data.append(fpr)
			tpr_data.append(tpr)
			precision_data.append(precision)
			recall_data.append(recall)
	
			pred = (df[thresh] > 0).tolist()
			gt = df['Ground Truth Label'].tolist()
	
			binary_precision.append(precision_score(gt, pred))
			binary_recall.append(recall_score(gt, pred))
			binary_f1.append(f1_score(gt, pred))
	
		thresh_prec_rec_df = pd.DataFrame(list(zip([str(p) for p in prob_thresh], binary_precision, binary_recall, binary_f1)), columns=['Thresh', 'Precision', 'Recall', 'F1'])
		thresh_prec_rec_melted_df = thresh_prec_rec_df.melt(id_vars=['Thresh'], value_vars=['Precision', 'Recall'])
	
		max_auc = max(auc_data)
		max_auc_idx = auc_data.index(max_auc)
		max_auc_data = [i for i, j in enumerate(auc_data) if j == max_auc]
		print('Probability: ' + str(prob_thresh[max_auc_data[0]]), 'AUC: ' + str(auc_data[max_auc_data[0]]))
	
		cutoff_prob.append(round(prob_thresh[max_auc_data[0]],6))
		cutoffs['tile_thresh'] = round(prob_thresh[max_auc_data[0]], 7)
		auc_probs['prob'] = auc_data
	
		auc_plotting['fpr'] = fpr_data[max_auc_idx]
		auc_plotting['tpr'] = tpr_data[max_auc_idx]
	
		print('-'*10)
	
		max_auprc = max(auprc_data)
		max_auprc_idx = auprc_data.index(max_auprc)
		max_auprc_data = [i for i, j in enumerate(auprc_data) if j == max_auprc]
		print('Probability: ' + str(prob_thresh[max_auprc_data[0]]), 'AUPRC: ' + str(auprc_data[max_auprc_data[0]]))
		
		auprc_cutoff_prob.append(round(prob_thresh[max_auprc_data[0]],6))
		auprc_cutoffs['tile_thresh'] = prob_thresh[max_auprc_data[0]]
		auprc_probs['prob'] = auprc_data
		auprc_plotting['precision'] = precision_data[max_auprc_idx]
		auprc_plotting['recall'] = recall_data[max_auprc_idx]

		pr_fig = precision_recall_plots(thresh_prec_rec_melted_df)
		pr_fig.savefig(os.path.join(output_path, 'pr_curve' + args.stain.upper() + '.png'))

		auc_fig = auc_thresh_plot(auc_probs, prob_thresh, args.stain)
		auc_fig.savefig(os.path.join(output_path, 'auc_prob_threshold' + args.stain.upper() + '.png'))

		roc_fig = roc_thresh_plot(cutoffs, auc_probs, auc_plotting, args.stain)
		roc_fig.savefig(os.path.join(output_path, 'roc_curve' + args.stain.upper() + '.png'))

		auprc_curve_fig = auprc_curve_plot(auprc_cutoffs, auprc_probs, auprc_plotting, args.stain)
		auprc_curve_fig.savefig(os.path.join(output_path, 'auprc_curve' + args.stain.upper() + '.png'))

		auprc_thresh_fig = auprc_thresh_plot(auprc_probs, prob_thresh, args.stain)
		auprc_thresh_fig.savefig(os.path.join(output_path, 'auprc_prob_threshold' + args.stain.upper() + '.png'))

	if args.vis:
		plt.show()