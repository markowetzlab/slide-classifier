# This file runs tile-level inference on the training, calibration and internal validation cohort
import argparse
import os
import time
import h5py
import numpy as np
import pandas as pd
import geojson
from tqdm import tqdm
import torch
import torch.nn as nn

from monai.data import DataLoader, CSVDataset, PatchWSIDataset
import monai.transforms as mt

from dataset_processing import class_parser
from models import get_network
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords

import warnings
warnings.filterwarnings('ignore')

def parse_args():
	parser = argparse.ArgumentParser(description='Run inference on slides.')

	parser.add_argument("--save_dir", default='results', help="path to folder where inference will be stored")

	#dataset processing
	parser.add_argument("--stain", choices=['he', 'qc', 'p53', 'tff3'], help="Stain to consider H&E (he), quality control (qc) or P53 (p53), TFF3(tff3)")
	parser.add_argument("--process_list", default=None, help="file containing slide-level ground truth to use.")
	parser.add_argument("--slide_path", default='slides', help="slides root folder")
	parser.add_argument("--format", default=".ndpi", help="extension of whole slide image")
	parser.add_argument("--reader", default='openslide', help="monai slide backend reader ('openslide' or 'cuCIM')")

	#model path and parameters
	parser.add_argument("--network", default='vgg_16', help="DL architecture to use")
	parser.add_argument("--model_path", required=True, help="path to stored model weights")

	parser.add_argument("--patch_path", default='patches', help="path to stored patch files")
	parser.add_argument("--patch_size", default=400, type=int, help="size of patches to extract")
	parser.add_argument("--patch_level", default=0, type=int, help="level to extract patches from")
	parser.add_argument("--input_size", default=None, type=int, help="size of tiles to extract")

	#data processing
	parser.add_argument("--batch_size", default=None, help="Batch size. Default is to use values set for architecture to run on 1 GPU.")
	parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to call for DataLoader")
	
	parser.add_argument("--he_threshold", default=0.99993, type=float, help="A threshold for detecting gastric cardia in H&E")
	parser.add_argument("--p53_threshold", default= 0.99, type=float, help="A threshold for Goblet cell detection in p53")
	parser.add_argument("--qc_threshold", default=0.99, help="A threshold for detecting gastric cardia in H&E")
	parser.add_argument("--tff3_threshold", default= 0.93, help="A threshold for Goblet cell detection in tff3")
	parser.add_argument("--lcp_cutoff", default=None, help='number of tiles to be considered low confidence positive')
	parser.add_argument("--hcp_cutoff", default=None, help='number of tiles to be considered high confidence positive')
	parser.add_argument("--impute", action='store_true', help="Assume missing data as negative")

	#class variables
	parser.add_argument("--dysplasia_separate", action='store_false', help="Flag whether to separate the atypia of uncertain significance and dysplasia classes")
	parser.add_argument("--respiratory_separate", action='store_false', help="Flag whether to separate the respiratory mucosa cilia and respiratory mucosa classes")
	parser.add_argument("--gastric_separate", action='store_false', help="Flag whether to separate the tickled up columnar and gastric cardia classes")
	parser.add_argument("--atypia_separate", action='store_false', help="Flag whether to perform the following class split: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other")
	parser.add_argument("--p53_separate", action='store_false', help="Flag whether to perform the following class split: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar")

	parser.add_argument("--ranked_class", default=None, help='particular class to rank tiles by')
	
	#outputs
	parser.add_argument("--xml", action='store_true', help='produce annotation files for ASAP in .xml format')
	parser.add_argument("--json", action='store_true', help='produce annotation files for QuPath in .geoJSON format')

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
	
	inference_dir = os.path.join(args.save_dir, 'inference')
	if not os.path.exists(inference_dir):
		os.makedirs(inference_dir, exist_ok=True)

	directories = {'source': args.slide_path, 
				   'save_dir': args.save_dir,
				   'patch_dir': args.patch_path, 
				   'inference_dir': inference_dir}

	if not args.silent:
		print("Outputting inference to: ", directories['save_dir'])

	slide_path = args.slide_path
	patch_size = args.patch_size
	network = args.network
	reader = args.reader

	if args.stain == 'he':
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
		channel_means = [0.7747305964175918, 0.7421753839460998, 0.7307385516144509]
		channel_stds = [0.2105364799974944, 0.2123423033814637, 0.20617556948731974]
	elif args.stain == 'qc':
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
		channel_means = [0.485, 0.456, 0.406]
		channel_stds = [0.229, 0.224, 0.225]
	elif args.stain == 'p53':
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
		channel_means = [0.7747305964175918, 0.7421753839460998, 0.7307385516144509]
		channel_stds = [0.2105364799974944, 0.2123423033814637, 0.20617556948731974]
	elif args.stain == 'tff3':
		file_name = 'TFF3'
		gt_label = 'TFF3 positive'
		classes = ['Equivocal', 'Negative', 'Positive']
		ranked_class = 'Positive'
		secondary_class = 'Equivocal'
		thresh = args.tff3_threshold
		mapping = {'Y': 1, 'N': 0}
		if args.lcp_cutoff is not None:
			lcp_triage_threshold = args.lcp_cutoff
		else:
			lcp_triage_threshold = 3
		if args.hcp_cutoff is not None:
			hcp_triage_threshold = args.hcp_cutoff
		else:
			hcp_triage_threshold = 40
		channel_means = [0.485, 0.456, 0.406]
		channel_stds = [0.229, 0.224, 0.225]
	else:
		raise AssertionError('args.stain must be he/qc, tff3, or p53.')

	if not args.silent:
		print('Channel Means: ', channel_means, '\nChannel Stds: ', channel_stds)

	trained_model, model_params = get_network(network, class_names=classes, pretrained=False)
	try:
		trained_model.load_state_dict(torch.load(args.model_path).module.state_dict())
	except: 
		trained_model = torch.load(args.model_path)
	
	# Modify the model to use the updated GELU activation function in later PyTorch versions 
	for name, module in trained_model.named_modules():
		if isinstance(module, nn.GELU):
			exec('trained_model.'+torchmodify(name)+'=nn.GELU()')

	# if args.multi_gpu:
		# trained_model = torch.nn.parallel.DistributedDataParallel(trained_model, device_ids=[args.local_rank], output_device=args.local_rank)

	# Use manual batch size if one has been specified
	if args.batch_size is not None:
		batch_size = args.batch_size
	else:
		batch_size = model_params['batch_size']
	
	if args.patch_size is not None:
		patch_size = args.patch_size
	else:
		patch_size = model_params['patch_size']

	if args.input_size is not None:
		input_size = args.patch_size
	else:
		input_size = model_params['patch_size']

	if torch.cuda.is_available() and (torch.version.hip or torch.version.cuda):
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	trained_model.to(device)
	trained_model.eval()

	eval_transforms = mt.Compose(
			[
				mt.Resized(keys="image", spatial_size=(input_size, input_size)),
				mt.ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=0.0, b_max=1.0),
				mt.ToTensord(keys=("image")),
				mt.TorchVisiond(keys=("image"), name="Normalize", mean=channel_means, std=channel_stds),
				mt.ToMetaTensord(keys=("image")),
			]
		)

	if args.process_list is not None:
		if os.path.isfile(args.process_list):
			#TODO read labels from csv to build process list
			process_list = pd.read_csv(args.process_list, index_col=0)
			process_list.dropna(subset=[file_name], inplace=True)
			if args.impute:
				process_list[gt_label] = process_list[gt_label].fillna('N')
			else:
				process_list.dropna(subset=[gt_label], inplace=True)
			process_list.sort_index(inplace=True)
			process_list[gt_label] = process_list[gt_label].map(mapping)
		else:
			raise AssertionError('Not a valid path for ground truth labels.')
	else:
		process_list = []
		for file in os.listdir(slide_path):
			if file.endswith(('.ndpi','.svs')):
				process_list.append(file)
		slides = sorted(process_list)
		process_list = pd.DataFrame(slides, columns=['slide_id'])
		process_list[gt_label] = 0
		process_list

	data_list = []

	for index, row in process_list.iterrows():
		slide = row['slide_id']
		slide_name = slide.replace(args.format, "")
		try:
			wsi = os.path.join(slide_path, slide)
		except:
			print(f'File {slide} not found.')
			continue

		if not os.path.exists(wsi):
			print(f'File {wsi} not found.')
			continue
		if not args.silent:
			print(f'\rProcessing case {index}/{len(process_list)}: ', end='')

		slide_output = os.path.join(directories['inference_dir'], slide_name)
		if os.path.isfile(slide_output + '.csv'):
			print(f'Inference for {slide_name} already exists.')
			tiles = pd.read_csv(slide_output+'.csv')
		else:
			patch_file = os.path.join(directories['patch_dir'], slide_name+'.h5')

			locations = pd.DataFrame(np.array(h5py.File(patch_file)['coords']), columns=['x_min','y_min'])
			locations['image'] = wsi
			print('Number of tiles:',len(locations))
			patch_locations = CSVDataset(locations,
								col_groups={"image": "image", "location": ["y_min", "x_min"]},
							)

			dataset = PatchWSIDataset(
				data=patch_locations,
				patch_size=patch_size,
				patch_level=args.patch_level,
				include_label=False,
				center_location=False,
				transform = eval_transforms,
				reader = reader
			)
			since = time.time()
			dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
			tile_predictions = []

			with torch.no_grad():
				print(f'\rCase {index}/{len(process_list)} {index} processing: ')
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
						prediction = {'x': tile_location[1], 'y': tile_location[0]}
						for i, pred in enumerate(preds):
							prediction[classes[i]] = pred
						tile_predictions.append(prediction)

			tiles = pd.DataFrame(tile_predictions)
			tiles.to_csv(slide_output+'.csv', index=False)

			time_elapsed = time.time() - since
			if not args.silent:
				print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

		counts = (tiles[ranked_class] > thresh).sum()
		data = {'CYT ID': index, 'Case Path': slide, str(ranked_class+' Tiles'): counts}
		data_list.append(data)
	
		if args.xml or args.json:
			positive = tiles[tiles[ranked_class] > thresh]

			annotation_path = os.path.join(directories['save_dir'], 'tile_annotations')
			os.makedirs(annotation_path, exist_ok=True)
			
			if len(positive) == 0:
				print(f'No annotations found at current threshold for {slide_name}')
			else:
				if args.xml:
					xml_path = os.path.join(annotation_path, slide_name+'_inference.xml')
					if not os.path.exists(xml_path):
						# Make ASAP file
						xml_header = """<?xml version="1.0"?><ASAP_Annotations>\t<Annotations>\n"""
						xml_tail =  f"""\t</Annotations>\t<AnnotationGroups>\t\t<Group Name="{ranked_class}" PartOfGroup="None" Color="#64FE2E">\t\t\t<Attributes />\t\t</Group>\t</AnnotationGroups></ASAP_Annotations>\n"""

						xml_annotations = ""
						for index, row in positive.iterrows():
							xml_annotations = (xml_annotations +
												"\t\t<Annotation Name=\""+str(row[ranked_class+'_probability'])+"\" Type=\"Polygon\" PartOfGroup=\""+ranked_class+"\" Color=\"#F4FA58\">\n" +
												"\t\t\t<Coordinates>\n" +
												"\t\t\t\t<Coordinate Order=\"0\" X=\""+str(row['x'])+"\" Y=\""+str(row['y'])+"\" />\n" +
												"\t\t\t\t<Coordinate Order=\"1\" X=\""+str(row['x']+patch_size)+"\" Y=\""+str(row['y'])+"\" />\n" +
												"\t\t\t\t<Coordinate Order=\"2\" X=\""+str(row['x']+patch_size)+"\" Y=\""+str(row['y']+patch_size)+"\" />\n" +
												"\t\t\t\t<Coordinate Order=\"3\" X=\""+str(row['x'])+"\" Y=\""+str(row['y']+patch_size)+"\" />\n" +
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
											[row['x']+patch_size, row['y']],
											[row['x']+patch_size, row['y']+patch_size],
											[row['x'], row['y']+patch_size],        
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
	
	df = pd.DataFrame.from_dict(data_list)
	df.to_csv(os.path.join(directories['save_dir'], 'process_list.csv'), index=False)
