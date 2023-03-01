import argparse
import os

from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
				classification_report, confusion_matrix)
from slidl.slide import Slide

from dataset_processing import class_parser

def parse_args():
	parser = argparse.ArgumentParser(description='Plot inference tiles')
	parser.add_argument("--description", default='triage', help="string to save results to.'")
	parser.add_argument("--labels", help="file containing slide-level ground truth to use.'")

	parser.add_argument("--from-file", help="genereate stats from existing run.'")

	parser.add_argument("--he_path", default=None, help="he slides root folder")
	parser.add_argument("--he_inference", default=None, help="path to directory containing atpyia inference file(s)")
	parser.add_argument("--he_threshold", help="A threshold above or equal to target tiles (atypia tiles for H&E, aberrant P53 columnar for P53)")

	parser.add_argument("--p53_path", default=None, help="p53 slides root folder")
	parser.add_argument("--p53_inference", default=None, help="path to directory containing p53 inference file(s)")
	parser.add_argument("--p53_threshold", help="A threshold above or equal to target tiles (atypia tiles for H&E, aberrant P53 columnar for P53)")
	parser.add_argument("--control", default=None, help='csv containing control tissue location.')

	parser.add_argument("--format", default=".ndpi", help="extension of whole slide image")
	parser.add_argument("--output", default='results', help="path to folder where inference maps will be stored")
	parser.add_argument("--csv", action='store_true', help="Output results to csv")

	#class variables
	parser.add_argument("--dysplasia_separate", action='store_false', help="Flag whether to separate the atypia of uncertain significance and dysplasia classes")
	parser.add_argument("--respiratory_separate", action='store_false', help="Flag whether to separate the respiratory mucosa cilia and respiratory mucosa classes")
	parser.add_argument("--gastric_separate", action='store_false', help="Flag whether to separate the tickled up columnar and gastric cardia classes")
	parser.add_argument("--atypia_separate", action='store_false', help="Flag whether to perform the following class split: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other")
	parser.add_argument("--p53_separate", action='store_false', help="Flag whether to perform the following class split: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar")

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()

	he_classes = class_parser('he', args.dysplasia_separate, args.respiratory_separate, args.gastric_separate, args.atypia_separate)
	he_root = args.he_path
	he_inference_root = args.he_inference
	if args.he_threshold is not None:
		he_threshold = float(args.he_threshold)
	else:
		he_threshold = 0.99

	atypia_hcn_triage_threshold = 0
	atypia_hcp_triage_threshold = 10

	p53_classes = class_parser('p53', args.p53_separate)
	p53_root = args.p53_path
	p53_inference_root = args.p53_inference
	if args.p53_threshold is not None:
		p53_threshold = float(args.p53_threshold)
	else:
		p53_threshold = 0.99

	p53_hcn_triage_threshold = 0
	p53_hcp_triage_threshold = 2

	output_path = args.output

	if args.from_file:
		df = pd.read_csv(args.from_file)
	else:
		if os.path.isfile(args.labels):
			labels = pd.read_csv(args.labels, index_col=0)
			labels.sort_index(inplace=True)
			labels['Atypia'] = labels['Atypia'].map(dict(Y=1, N=0))
			labels['P53 positive'] = labels['P53 positive'].fillna('N').map(dict(Y=1, N=0))
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
			
			if he_inference_root is not None:
				ranked_class = 'atypia'
				
				he_file = row['H&E']
				he_inference = he_file.replace(args.format, '_inference.pml')

				he_slide = os.path.join(he_root, he_file)
				he_inference = os.path.join(he_inference_root, he_inference)

				he_slide = Slide(he_inference, newSlideFilePath=he_slide)
				he_tiles = he_slide.numTilesAboveClassPredictionThreshold(ranked_class, he_threshold)
				
				if he_tiles <= atypia_hcn_triage_threshold:
					he_triage = 'hcn'
				elif he_tiles >= atypia_hcp_triage_threshold:
					he_triage = 'hcp'
				else:
					he_triage = 'lcp'
				
				case.update({'H&E Case': he_file, 'Atypia GT': row['Atypia'], 'Atypia Tiles': he_tiles, 'Atypia Triage': he_triage})
			
			if p53_inference_root is not None:
				ranked_class = 'aberrant_positive_columnar'
				
				p53_file = row['P53']
				p53_inference = p53_file.replace(args.format, '_inference.pml')

				p53_slide = os.path.join(p53_root, p53_file)
				p53_inference = os.path.join(p53_inference_root, p53_inference)

				p53_slide = Slide(p53_inference, newSlideFilePath=p53_slide)

				X_tiles = p53_slide.numTilesInX
				Y_tiles = p53_slide.numTilesInY
				X_control = round(X_tiles/4)
				Y_control = round(Y_tiles/4)
				
				if args.control is not None:
					control_loc = controls.loc[index]
					p53_tiles = 0
					for tile_address, tile_entry in p53_slide.tileDictionary.items():
						if tile_entry['classifierInferencePrediction'][ranked_class] >= p53_threshold:
							if tile_address[0] < X_control and control_loc['L'] == 1:
								continue
							elif X_tiles - X_control < tile_address[0] and control_loc['R'] == 1:
								continue
							elif Y_tiles - Y_control < tile_address[1] and (control_loc['B'] == 1 and control_loc['L'] == 0 and control_loc['R'] == 0):
								continue
							elif tile_address[1] < Y_control and (control_loc['T'] == 1 and control_loc['L'] == 0 and control_loc['R'] == 0):
								continue
							else:
								p53_tiles += 1
				else:
					#P53 slides in Delta contain a positive control tissue which skews analysis
					control_loc = {'CYT ID': index, 'L':0, 'R':0, 'T':0, 'B':0}

					vertical_middle = 0
					tiles_left = 0
					tiles_right = 0

					horizontal_middle = 0
					tiles_top = 0
					tiles_bottom = 0
					
					for tile_address, tile_entry in p53_slide.tileDictionary.items():
						if p53_slide.tileDictionary[tile_address]['classifierInferencePrediction'][ranked_class] >= p53_threshold:
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

					p53_tiles = p53_slide.numTilesAboveClassPredictionThreshold(ranked_class, p53_threshold)

				if p53_tiles <= p53_hcn_triage_threshold:
					p53_triage = 'hcn'
				elif p53_tiles > p53_hcp_triage_threshold:
					p53_triage = 'hcp'
				else:
					p53_triage = 'lcp'
				
				case.update({'P53 Case': p53_file, 'P53 GT': row['P53 positive'], 'P53 Tiles': p53_tiles, 'P53 Triage': p53_triage})

			if he_inference_root is not None and p53_inference_root is not None:
				if row['Atypia'] == 1 or row['P53 positive'] == 1:
					gt = 1
				else:
					gt = 0
				case.update({'Pathologist GT': gt})

				if he_triage == 'hcp' and p53_triage == 'hcp':
					case.update({'Triage Class': 'A'})
				elif he_triage == 'hcp' and p53_triage == 'lcp':
					case.update({'Traige Class': 'B'})
				elif he_triage == 'hcp' and p53_triage == 'hcn':
					case.update({'Traige Class': 'C'})
				elif he_triage == 'lcp' and p53_triage == 'hcp':
					case.update({'Traige Class': 'D'})
				elif he_triage == 'lcp' and p53_triage == 'lcp':
					case.update({'Traige Class': 'E'})
				elif he_triage == 'lcp' and p53_triage == 'hcn':
					case.update({'Traige Class': 'F'})
				elif he_triage == 'hcn' and p53_triage == 'hcp':
					case.update({'Traige Class': 'G'})
				elif he_triage == 'hcn' and p53_triage == 'lcp':
					case.update({'Traige Class': 'H'})
				elif he_triage == 'hcn' and p53_triage == 'hcn':
					case.update({'Traige Class': 'I'})
				else:
					raise Warning('Row does not fit into a triage class letter.')

			results.append(case)

		df = pd.DataFrame.from_dict(results)

	if he_inference_root is not None:
		df['Atypia Results'] = (df['Atypia Tiles'] > 0).astype(int)
		results = df['Atypia Results'].astype(int).tolist()
		gt = df['Atypia GT'].astype(int).tolist()

		tn, fp, fn, tp = confusion_matrix(gt, results).ravel()
		auc = roc_auc_score(gt, results)
		precision = precision_score(gt, results)
		recall = recall_score(gt, results)
		f1 = f1_score(gt, results)

		# Evaluate model's performance relative to pathologist ground truth
		print('Atypia model vs. pathologist ground truth\n')
		print('AUC: ', auc)
		print('Sensitivity: ', recall)
		print('Specificity: ', tn/(tn+fp))
		print('Precision: ', precision)
		print('F1: ', f1, '\n')
		print(confusion_matrix(gt, results))

		print(classification_report(gt, results))

	if p53_inference_root is not None:
		if args.csv:
			control_df = pd.DataFrame.from_dict(controls)
			control_df.to_csv(os.path.join(p53_root, 'controls.csv'), index=False)
		df['P53 Results'] = (df['P53 Tiles'] > 0).astype(int)
		results = df['P53 Results'].astype(int).tolist()
		gt = df['P53 GT'].astype(int).tolist()
		
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
		print(confusion_matrix(gt, results))
		
		print(classification_report(gt, results))

	if he_inference_root is not None and p53_inference_root is not None:
		df.loc[df['Atypia Results'] >= 1 or df['P53 Results'] >= 1, 'Results'] =  1
		df.loc[df['Atypia Results'] == 0 and df['P53 Results'] == 0, 'Results'] = 0

		print('Atypia-P53 model vs. pre-AI pathologist ground truth')
		print(classification_report(df['Pathologist GT'].astype(int).tolist(), df['Results'].astype(int).tolist()))
		print(confusion_matrix(df['Pathologist GT'].astype(int).tolist(), df['Results'].astype(int).tolist()))

	if args.csv:
		# Emit CSVs of these datasets
		df.to_csv(os.path.join(output_path, args.description + '_results.csv'), index=False)

