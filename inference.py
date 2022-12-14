# Marcel Gehrung and Adam Berman
# This file runs tile-level inference on the training, calibration and internal validation cohort

import argparse
import glob
import os
import pickle
import time

import torch
import torch.nn as nn
#sys.path.append('/home/cri.camres.org/berman01/pathml') # install from here: https://github.com/9xg/pathml
from pathml.pathml import slide
from tqdm import tqdm

from dataset_processing import class_parser
from dataset_processing.best import WholeSlideImageDataset
from dataset_processing.image import channel_averages, image_transforms
from models import get_network


def parse_args():
	parser = argparse.ArgumentParser(description='Run inference on slides.')
	
	#biomarkers
	parser.add_argument("--stain", required=True, help="he or p53")
	
	#model path and parameters
	parser.add_argument("--network", required=True, help="which CNN architecture to use")
	parser.add_argument("--model_path", required=True, help="path to stored model weights")
	parser.add_argument("--output", required=True, help="path to folder where inference maps will be stored")

	parser.add_argument("--tile_size", required=True, help="architecture tile size")
	parser.add_argument("--overlap", default=0, help="what fraction of the tile edge neighboring tiles should overlap horizontally and vertically (default is 0)")
	parser.add_argument("--foreground_only", action='store_true', help="Foreground with tissue only")

	#data paths
	parser.add_argument("--slide_path", required=True, help="slides root folder")
	parser.add_argument("--format", default="ndpi", help="extension of whole slide image without full stop")

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
	
	parser.add_argument('--silent', action='store_true', help='Flag which silences tqdm on servers')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	slide_path = args.slide_path

	channel_means, channel_stds = channel_averages(slide_path, args.channelmeans, args.channelstds)
	
	network = args.network
	classes = class_parser(args.stain, args.dysplasia_separate, args.respiratory_separate, args.gastric_separate, args.atypia_separate, args.p53_separate)
	trained_model, params = get_network(network, class_names=classes)
	
	# Use manual batch size if one has been specified
	if args.batch_size is not None:
		batch_size = args.batch_size
	else:
		batch_size = params['batch_size']
	patch_size = params['patch_size']

	trained_model.load_state_dict(torch.load(args.model_path))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		print("Inferring on GPU")
	else:
		print("Inferring on CPU")

	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		trained_model = nn.DataParallel(trained_model)

	trained_model.to(device)
	trained_model.eval()

	cases = []
	case_paths = glob.glob(os.path.join(args.slidesrootfolder, '*.'+args.wsiextension))

	output_path = args.output
	if not os.path.exists(output_path):
		os.makedirs(output_path, exist_ok=True)
	print("Outputting inference to: ", output_path)
	pathTileSize = 400

	data_transforms = image_transforms(channel_means, channel_stds, patch_size)['val']

	for case in tqdm(cases, disable=args.silent):
		if os.path.isfile(os.path.join(output_path, case + '_inference.pickle')):
			print("Case already processed. Skipping...")
			continue

		pathSlide = slide.Slide(os.path.join(slide_path, case + '.' + args.format))
		pathSlide.setTileProperties(tileSize=pathTileSize, overlap=float(args.overlap))

		if args.foreground_only:
			pathSlide.detectForeground(threshold=95)

		dataset = WholeSlideImageDataset(pathSlide, foreground_only=args.foreground_only, transform=data_transforms)

		since = time.time()
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)
		
		with torch.no_grad():
			for inputs in tqdm(dataloader):
				inputTile = inputs['image'].to(device)
				output = trained_model(inputTile)
				output = output.to(device)

				batch_prediction = torch.nn.functional.softmax(output, dim=1).cpu().data.numpy()

				for index in range(len(inputTile)):
					tileAddress = (inputs['tileAddress'][0][index].item(), inputs['tileAddress'][1][index].item())
					pathSlide.appendTag(tileAddress, 'prediction', batch_prediction[index, ...])
		
		tileDictionaryWithInference = {'maskSize': (pathSlide.numTilesInX, pathSlide.numTilesInY), 'tileDictionary': pathSlide.tileDictionary}
		pickle.dump(tileDictionaryWithInference, open(os.path.join(output_path, case + '_inference.pickle'), 'wb'))
		time_elapsed = time.time() - since
		print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
