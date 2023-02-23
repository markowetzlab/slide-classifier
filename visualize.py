import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvips as pv
from slidl.slide import Slide
from utils.visualisation.display_slide import slide_image

from dataset_processing import class_parser

def parse_args():
	parser = argparse.ArgumentParser(description='Plot inference tiles')
	parser.add_argument("--description", default='triage', help="string to save results to.")
	parser.add_argument("--stain", required=True, help="he or p53")
	parser.add_argument("--labels", help="file containing slide-level ground truth to use.'")
	parser.add_argument("--target", type=str, default=None, help="Target class to identify, if None then defaults to stain class.")
	parser.add_argument("--gt", type=str, default=None, help="Column containing ground truth labels")

	parser.add_argument("--slide_path", required=True, help="slides root folder")
	parser.add_argument("--format", default=".ndpi", help="extension of whole slide image")
	parser.add_argument("--inference", required=True, help="path to directory containing inference file(s)")
	parser.add_argument("--output", required=True, help="path to folder where inference maps will be stored")
	
	#class variables
	parser.add_argument("--dysplasia_separate", action='store_false', help="Flag whether to separate the atypia of uncertain significance and dysplasia classes")
	parser.add_argument("--respiratory_separate", action='store_false', help="Flag whether to separate the respiratory mucosa cilia and respiratory mucosa classes")
	parser.add_argument("--gastric_separate", action='store_false', help="Flag whether to separate the tickled up columnar and gastric cardia classes")
	parser.add_argument("--atypia_separate", action='store_false', help="Flag whether to perform the following class split: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other")
	parser.add_argument("--p53_separate", action='store_false', help="Flag whether to perform the following class split: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar")

	parser.add_argument("--extract", default=None, help="A threshold above or equal to target tiles (atypia tiles for H&E, aberrant P53 columnar for P53) are extracted to the output folder. Default is not to extract these tiles.")
	parser.add_argument("--tiles", action='store_true', help='save tile images')
	parser.add_argument("--vis", action='store_true', help='save heatmaps')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()

	stain = args.stain
	target = args.target

	slide_path = args.slide_path
	inference_path = args.inference
	output_path = args.output

	if target is not None:
		ranked_class = target
	elif stain == 'he':
		ranked_class = 'atypia'
		ranked_label = 'Atypia'
	elif stain == 'p53':
		ranked_class = 'aberrant_positive_columnar'
		ranked_label = 'P53 positive'
		controls = pd.read_csv(os.path.join(inference_path, 'controls.csv'))
	elif stain == 'tff3':
		ranked_class = 'positive'
		ranked_label = 'TFF3 positive'
	else:
		raise AssertionError('Stain currently must be he/p53/tff3.')

	classes = class_parser(stain, args.dysplasia_separate, args.respiratory_separate, args.gastric_separate, args.atypia_separate, args.p53_separate)

	#Prediction csv from output of inference.py
	if os.path.isfile(args.labels):
		labels = pd.read_csv(args.labels, index_col='CYT ID')
		labels.sort_index(inplace=True)
	else:
		raise AssertionError('Not a valid path for ground truth labels.')

	for index, row in labels.iterrows():
		slide_name = row['Case Path'].replace(args.format, "")
		inference_file = slide_name + '_inference.pml'

		slide_file = os.path.join(slide_path, slide_name + args.format)
		case_inference = os.path.join(inference_path, inference_file)

		slidl_slide = Slide(case_inference, newSlideFilePath=slide_file)
		thumbnail = slidl_slide.thumbnail(level=4)

		if args.extract:
			slide_im = pv.Image.new_from_file(slide_file, level=0)

		class_masks = {}
		class_masks[ranked_class] = np.zeros((slidl_slide.numTilesInX, slidl_slide.numTilesInY)[::-1])
		ranked_dict = {}

		if args.stain == 'p53':
			X_tiles = slidl_slide.numTilesInX
			Y_tiles = slidl_slide.numTilesInY
			X_control = round(X_tiles/4)
			Y_control = round(Y_tiles/4)
			control_loc = controls.loc[index]

		for tile_address,tile_entry in slidl_slide.tileDictionary.items():
			for class_name in classes:
				# extract target tiles
				if args.extract:
					if class_name == ranked_class:
						if args.stain == 'p53':
							prob = tile_entry['classifierInferencePrediction'][class_name] if 'classifierInferencePrediction' in tile_entry else 0
							if prob >= float(args.extract): # tile is considered target
								if tile_address[0] < X_control and control_loc['L'] == 1:
									continue
								elif X_tiles - X_control < tile_address[0] and control_loc['R'] == 1:
									continue
								elif Y_tiles - Y_control < tile_address[1] and (control_loc['B'] == 1 and control_loc['L'] == 0 and control_loc['R'] == 0):
									continue
								elif tile_address[1] < Y_control and (control_loc['T'] == 1 and control_loc['L'] == 0 and control_loc['R'] == 0):
									continue
								else:
									# SAVE TARGET TILE ASAP ANNOTATION FILE
									ranked_dict[str(prob)+'_'+str(tile_entry['x'])+'_'+str(tile_entry['y'])] = {ranked_class+'_probability': prob, 'x': tile_entry['x'], 'y': tile_entry['y'], 'width': tile_entry['width']}
						else:
							prob = tile_entry['classifierInferencePrediction'][class_name] if 'classifierInferencePrediction' in tile_entry else 0
							if prob >= float(args.extract): # tile is considered target
								# SAVE TARGET TILE ASAP ANNOTATION FILE
								ranked_dict[str(prob)+'_'+str(tile_entry['x'])+'_'+str(tile_entry['y'])] = {ranked_class+'_probability': prob, 'x': tile_entry['x'], 'y': tile_entry['y'], 'width': tile_entry['width']}
	
								if args.tiles:
									# SAVE TARGET TILES
									if not os.path.exists(os.path.join(output_path, ranked_class+'_tiles', slide_file, ranked_class + '_' + str(prob) + '_' + str(
										tile_entry['x']) + '_' + str(tile_entry['y']) + '_' + str(tile_entry['width']) + '.jpg')):
										try:
											target_area = slide_im.extract_area(
												tile_entry['x'], tile_entry['y'], tile_entry['width'], tile_entry['height'])
											os.makedirs(os.path.join(output_path, ranked_class+'_tiles', slide_name), exist_ok=True)
											target_area.write_to_file(os.path.join(output_path, ranked_class+'_tiles', slide_name, ranked_class + '_' + str(prob) + '_' + str(
												tile_entry['x']) + '_' + str(tile_entry['y']) + '_' + str(tile_entry['width']) + '.jpg'), Q=100)
											print('Made target tile')
										except:
											print('Skipping tile that goes beyond the edge of the WSI...')
									else:
										print('target tile already exists, skipping...')

				if ranked_class:
					if (class_name == ranked_class):
						class_masks[class_name][tile_address[1],tile_address[0]] = tile_entry['classifierInferencePrediction'][class_name] if 'classifierInferencePrediction' in tile_entry else 0
				else:
					class_masks[class_name][tile_address[1],tile_address[0]] = tile_entry['classifierInferencePrediction'][class_name] if 'classifierInferencePrediction' in tile_entry else 0

		# Make ASAP file
		xml_header = """<?xml version="1.0"?><ASAP_Annotations>\t<Annotations>\n"""
		xml_tail = 	"""\t</Annotations>\t<AnnotationGroups>\t\t<Group Name="atypia" PartOfGroup="None" Color="#64FE2E">\t\t\t<Attributes />\t\t</Group>\t</AnnotationGroups></ASAP_Annotations>\n"""
		xml_tail = xml_tail.replace('atypia', ranked_class)

		xml_annotations = ""

		if ranked_dict:
			annotation_path = os.path.join(output_path, args.description + '_' + ranked_class + '_tile_annotations')
			if not os.path.exists(os.path.join(annotation_path, slide_name+'_inference_'+ranked_class+'.xml')):
				os.makedirs(annotation_path, exist_ok=True)
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
				with open(os.path.join(annotation_path, slide_name+'_inference_'+ranked_class+'.xml'), "w") as annotation_file:
					annotation_file.write(xml_header + xml_annotations + xml_tail)
			else:
				print('Automated annotation file already exists...')
		else:
			print(f'No tiles found {index}.')

		if args.vis:
			slide_im = slide_image(slidl_slide, stain, classes)
			im_path = os.path.join(output_path, 'images')
			if not os.path.exists(im_path):
				os.makedirs(im_path)
			slide_im.plot_thumbnail(case_id=index, target=ranked_class)
			slide_im.draw_class(target = ranked_class, threshold=args.extract)
			slide_im.plot_class(target = ranked_class)
			slide_im.save(im_path, slide_file.replace(args.format, "_" + ranked_class + args.extract))
			plt.show()
			plt.close('all')