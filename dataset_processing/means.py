#Generate means and stds for image normalisation from slide root folder

import glob
import os
import pickle
import argparse
import random
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from slidl.slide import Slide

def parse_args():
	parser = argparse.ArgumentParser(description='Create tile dictionaries from annotation ROIs.')

	parser.add_argument("--slides_path", required=True, help="slides root folder")
	parser.add_argument("--tiles", default=500, help="Number of random tiles to extract")
	parser.add_argument("--format", default=".ndpi", help="extension of whole slide image without full stop")

	parser.add_argument("--silent", action="store_true", help="Silence tqdm on servers")
	
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()

	verbose = True
	magnificationLevel = 0  # 1 = 20(x) or 0 = 40(x)
	slidesRootFolder = args.slides_path

	cases = glob.glob(os.path.join(args.slides_path, '*' + args.format))
	cases.sort()
	num_samples = len(cases)
	print('Number of files:', num_samples)

	format_to_dtype = {
		'uchar': np.uint8,
		'char': np.int8,
		'ushort': np.uint16,
		'short': np.int16,
		'uint': np.uint32,
		'int': np.int32,
		'float': np.float32,
		'double': np.float64,
		'complex': np.complex64,
		'dpcomplex': np.complex128,
	}

	channel_sums = np.zeros(3)
	channel_squared_sums = np.zeros(3)
	tile_counter = 0
	tile_size = 400
	normalise = transforms.Compose([transforms.ToTensor()])

	for case in tqdm(cases, disable=args.silent):
		slidl_slide = Slide(case).setTileProperties(tileSize=tile_size, tileOverlap=0)

		ta = []
		for address in slidl_slide.suitableTileAddresses():
			ta.append(address)
		ta = random.sample(ta, args.tiles)

		for tl in ta:
			tile_counter += 1
			nparea = slidl_slide.getTile(tl, writeToNumpy=True)[...,:3]
			nparea = normalise(nparea).numpy()

			local_channel_sums = np.sum(nparea, axis=(1,2))
			local_channel_squared_sums = np.sum(np.square(nparea), axis=(1,2))
	
			channel_sums = np.add(channel_sums, local_channel_sums)
			channel_squared_sums = np.add(channel_squared_sums, local_channel_squared_sums)

	# save channel means and stds
	total_pixels_per_channel = tile_counter * tile_size * tile_size
	channel_means = np.divide(channel_sums, total_pixels_per_channel)
	channel_squared_means = np.divide(channel_squared_sums, total_pixels_per_channel)
	channel_variances = np.subtract(channel_squared_means, np.square(channel_means))
	channel_stds = np.sqrt(channel_variances * (total_pixels_per_channel / (total_pixels_per_channel-1)))
	means_and_stds = {'channel_means': channel_means.tolist(), 'channel_stds': channel_stds.tolist()}
	with open('channel_means_and_stds.pickle', 'wb') as f:
		pickle.dump(means_and_stds,	f)

	print('Channel means:', channel_means.tolist())
	print('Channel standard deviations:', channel_stds.tolist())