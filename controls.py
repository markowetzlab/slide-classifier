import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
				classification_report, confusion_matrix)
from slidl.slide import Slide

from dataset_processing import class_parser

def parse_args():
	parser = argparse.ArgumentParser(description='Plot inference tiles')
	parser.add_argument("--description", default='triage', help="string to save results to.'")

	parser.add_argument("--stain", choices=['p53', 'tff3'], help="slide stain p53 or tff3")

	parser.add_argument("--from-file", help="generate stats from existing run.'")
	parser.add_argument("--labels", help="file containing slide-level ground truth to use.'")
	parser.add_argument("--impute", action='store_true', help="Assume missing data as negative")

	parser.add_argument("--path", default=None, help="p53 slides root folder")
	parser.add_argument("--inference", default=None, help="path to directory containing p53 inference file(s)")
	parser.add_argument("--threshold",  default=None, help="A threshold above or equal to target tiles (atypia tiles for H&E, aberrant P53 columnar for P53)")

	parser.add_argument("--format", default=".ndpi", help="extension of whole slide image")
	parser.add_argument("--output", default='results', help="path to folder where inference maps will be stored")

	parser.add_argument("--control", default=None, help='csv containing control tissue location.')
	parser.add_argument("--csv", action='store_true', help="Output results to csv")

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()

	root = args.path
	inference_root = args.inference

	if args.stain == 'p53':
		case_path = 'P53'
		label = 'P53 positive'
		classes = class_parser('p53')
		if args.threshold is not None:
			threshold = float(args.threshold)
		else:
			threshold = 0.99

		ranked_class = 'aberrant_positive_columnar'

		inference_suffix = '_inference.pml'

		hcn_triage_threshold = 0
		hcp_triage_threshold = 2

	if args.stain == 'tff3':
		case_path = 'TFF3'
		label = 'TFF3 positive'
		classes = ['Equivocal', 'Negative', 'Positive']
		if args.threshold is not None:
			threshold = float(args.threshold)
		else:
			threshold = 0.93
		
		ranked_class = 'Positive'

		inference_suffix = '_triage.pml'

		hcn_triage_threshold = 3
		hcp_triage_threshold = 40

	output_path = args.output

	if args.from_file:
		df = pd.read_csv(args.from_file)
	else:
		if os.path.isfile(args.labels):
			labels = pd.read_csv(args.labels, index_col=0)
			labels.sort_index(inplace=True)
			if args.impute:
				labels[label] = labels[label].fillna('N').map(dict(Y=1, N=0))
			else:
				labels.dropna(subset=[label], inplace=True)
				labels[label] = labels[label].map(dict(Y=1, N=0))
		else:
			raise AssertionError('Not a valid path for ground truth labels.')

		columns=['CYT ID', 'L', 'R', 'T', 'B']
		controls = []
		if args.control is not None:
			controls = pd.read_csv(args.control, index_col='CYT ID')

		results = []
		process = 0
		for index, row in labels.iterrows():
			process += 1
			print('Processing case: ', index, f'({process}/{len(labels)})\r', end="")
			case = {'CYT ID': index}

			try:
				file = row[case_path]
				inference_file = file.replace(args.format, inference_suffix)
			except:
				continue

			case_slide = os.path.join(root, file)
			inference_path = os.path.join(inference_root, inference_file)

			slide = Slide(inference_path, newSlideFilePath=case_slide)

			X_tiles = slide.numTilesInX
			Y_tiles = slide.numTilesInY
			X_control = round(X_tiles/4)
			Y_control = round(Y_tiles/4)

			if args.control is not None:
				control_loc = controls.loc[index]
				tiles = 0
				for tile_address, tile_entry in slide.tileDictionary.items():
					if tile_entry['classifierInferencePrediction'][ranked_class] >= threshold:
						if tile_address[0] < X_control and control_loc['L'] == 1:
							continue
						elif X_tiles - X_control < tile_address[0] and control_loc['R'] == 1:
							continue
						elif Y_tiles - Y_control < tile_address[1] and (control_loc['B'] == 1 and (control_loc['L'] == 0 or control_loc['R'] == 0)):
							continue
						elif tile_address[1] < Y_control and (control_loc['T'] == 1 and (control_loc['L'] == 0 or control_loc['R'] == 0)):
							continue
						else:
							tiles += 1
			else:
				#P53 slides in Delta contain a positive control tissue which skews analysis
				control_loc = {'CYT ID': index, 'L':0, 'R':0, 'T':0, 'B':0}

				vertical_middle = 0
				tiles_left = 0
				tiles_right = 0

				horizontal_middle = 0
				tiles_top = 0
				tiles_bottom = 0

				for tile_address, tile_entry in slide.tileDictionary.items():
					if slide.tileDictionary[tile_address]['classifierInferencePrediction'][ranked_class] >= threshold:
						if tile_address[0] < X_control:
							tiles_left += 1
						elif X_tiles - X_control < tile_address[0]:
							tiles_right += 1
						else:
							vertical_middle += 1

						if tile_address[1] < Y_control:
							tiles_top += 1
						elif Y_tiles - Y_control < tile_address[1]:
							tiles_bottom += 1
						else:
							horizontal_middle += 1

				#Locate the control tissue
				loc = ''
				if tiles_top > horizontal_middle + tiles_bottom:
					control_loc['T'] = 1
					loc += 'T'
				if tiles_bottom > horizontal_middle + tiles_top:
					control_loc['B'] = 1
					loc += 'B'
				if tiles_left > vertical_middle + tiles_right:
					control_loc['L'] = 1
					loc += 'L'
				if tiles_right > vertical_middle + tiles_left:
					control_loc['R'] = 1
					loc += 'R'
				control_loc['Loc'] = loc
				controls.append(control_loc)

				tiles = slide.numTilesAboveClassPredictionThreshold(ranked_class, threshold)

			if tiles <= hcn_triage_threshold:
				triage = 'hcn'
			elif tiles > hcp_triage_threshold:
				triage = 'hcp'
			else:
				triage = 'lcp'

			case.update({'Case': file, 'GT': row[label], 'Tiles': tiles, 'Triage': triage})

			results.append(case)

		df = pd.DataFrame.from_dict(results)

	if args.csv:
		control_df = pd.DataFrame.from_dict(controls)
		control_path = os.path.join(args.output, args.description+'.csv')
		control_df.to_csv(control_path, index=False)
		print(f'Controls saved to {control_path}')

	df['Results'] = (df['Tiles'] > hcn_triage_threshold).astype(int)
	results = df['Results'].astype(int).tolist()
	gt = df['GT'].astype(int).tolist()

	tn, fp, fn, tp = confusion_matrix(gt, results).ravel()
	auc = roc_auc_score(gt, results)
	precision = precision_score(gt, results)
	recall = recall_score(gt, results)
	f1 = f1_score(gt, results)

	print('P53 model vs. pathologist ground truth\n')
	print('AUC: ', auc)
	print('Sensitivity: ', recall)
	print('Specificity: ', tn/(tn+fp))
	print('Precision: ', precision)
	print('F1: ', f1, '\n')

	print(f'CM\tGT Positive\tGT Negative')
	print(f'Pred Positive\t{tp}\t{fp}')
	print(f'Pred Negative\t{fn}\t{tn}')

	print(classification_report(gt, results))

	if args.csv:
		# Emit CSVs of these datasets
		df.to_csv(os.path.join(output_path, args.description + '_results.csv'), index=False)
