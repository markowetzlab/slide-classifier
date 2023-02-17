import argparse
import os

import pandas as pd
from sklearn.metrics import classification_report
from slidl.slide import Slide

from dataset_processing import class_parser


def parse_args():
	parser = argparse.ArgumentParser(description='Plot inference tiles')
	parser.add_argument("--labels", help="file containing slide-level ground truth to use.'")

	parser.add_argument("--he_path", default=None, help="he slides root folder")
	parser.add_argument("--he_inference", default=None, help="path to directory containing atpyia inference file(s)")
	parser.add_argument("--he_threshold", help="A threshold above or equal to target tiles (atypia tiles for H&E, aberrant P53 columnar for P53)")

	parser.add_argument("--p53_path", default=None, help="p53 slides root folder")
	parser.add_argument("--p53_inference", default=None, help="path to directory containing p53 inference file(s)")
	parser.add_argument("--p53_threshold", help="A threshold above or equal to target tiles (atypia tiles for H&E, aberrant P53 columnar for P53)")

	parser.add_argument("--format", default=".ndpi", help="extension of whole slide image")
	parser.add_argument("--output", default='results', help="path to folder where inference maps will be stored")

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
	if args.p53_threshold is not None:
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

	if os.path.isfile(args.labels):
		labels = pd.read_csv(args.labels, index_col=0)
		labels.sort_index(inplace=True)
		labels['Atypia'] = labels['Atypia'].map(dict(Y=1, N=0))
		labels['P53 positive'] = labels['P53 positive'].map(dict(Y=1, N=0))
	else:
		raise AssertionError('Not a valid path for ground truth labels.')

	results = []
	for index, row in labels.iterrows():
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
			elif he_tiles < atypia_hcp_triage_threshold:
				he_triage = 'hcp'
			else:
				he_triage = 'lcp'
			
			case.update({'Atypia GT': row['Atypia'], 'Atypia Tiles': he_tiles, 'Atypia Triage': he_triage})
		
		if p53_inference_root is not None:

			ranked_class = 'atypia'
			
			p53_file = row['H&E']
			p53_inference = p53_file.replace(args.format, '_inference.pml')

			p53_slide = os.path.join(p53_root, p53_file)
			p53_inference = os.path.join(p53_inference_root, p53_inference)

			p53_slide = Slide(p53_inference, newSlideFilePath=p53_slide)
			p53_tiles = p53_slide.numTilesAboveClassPredictionThreshold(ranked_class, p53_threshold)
			
			if p53_tiles <= p53_hcn_triage_threshold:
				p53_triage = 'hcn'
			elif p53_tiles < p53_hcp_triage_threshold:
				p53_triage = 'hcp'
			else:
				p53_triage = 'lcp'
			
			case.update({'P53 GT': row['Atypia'], 'P53 Tiles': p53_tiles, 'P53 Triage': p53_triage})

		if he_inference_root is not None and p53_inference_root is not None:
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


	# # Evaluate model's performance relative to pathologist ground truth
	# print('Atypia model vs. pathologist ground truth')
	# print(classification_report(atypia_results['atypia_ground_truth_no_ai_review'], atypia_results['atypia_tile_count_greater_than_0']))

	# print('P53 val model vs. pre-AI pathologist ground truth')
	# print(classification_report(p53_results['p53_ground_truth_no_ai_review'], p53_results['p53_tile_count_greater_than_0']))

	# print('Atypia-P53 model vs. pre-AI pathologist ground truth')
	# print(classification_report(atypiap53_results['atypiap53_ground_truth_no_ai_review'], atypiap53_results['atypiap53_tile_count_greater_than_0']))

	# Emit CSVs of these datasets
	df = pd.DataFrame.from_dict(results)
	df.to_csv('', index_label='CYT ID')

