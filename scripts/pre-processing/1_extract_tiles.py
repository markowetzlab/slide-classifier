# 1_extractTilesFromDictionaries.py
#
# Adam Berman and Marcel Gehrung
#
# This file checks takes tile-dictionaries and generates individual tiles as results
#

import glob
import os
import pickle
import argparse
import numpy as np
import pyvips as pv
from tqdm import tqdm
from torchvision import transforms

parser = argparse.ArgumentParser(description='Create tile dictionaries from annotation ROIs.')
parser.add_argument("-slidesrootfolder", required=True, help="slides root folder")
parser.add_argument("-tiledictionariesrootfolder", required=True, help="tile dictionaries root folder")
parser.add_argument("-output", required=True, help="folder to store newly extracted tiles in")
parser.add_argument("-wsiextension", default="ndpi", help="extension of whole slide image without full stop")
args = parser.parse_args()
print(args)

verbose = True
magnificationLevel = 0  # 1 = 20(x) or 0 = 40(x)
slidesRootFolder = args.slidesrootfolder #"/media/berman01/Backup Plus"
tileDictionariesFileList = glob.glob(os.path.join(args.tiledictionariesrootfolder, '*.p')) #glob.glob("data/tileDictionaries-he-400px-at-40x/*.p")
tileDictionariesFileList.sort()
tilesRootFolder = args.output #"/media/berman01/Backup Plus/atypia-detection-tiles"
wsi_extension = args.wsiextension

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
tile_size = 0
normalize_to_1max = transforms.Compose([transforms.ToTensor()])

#print(tileDictionariesFileList)

for tileDictionaryFile in tqdm(tileDictionariesFileList, desc="Processing tile dictionaries"):
    if verbose:
        print('Opening file: ' + tileDictionaryFile)
    tileDictionary = pickle.load(open(tileDictionaryFile, "rb"))
    slideFileName = os.path.split(tileDictionaryFile)[-1].replace(".p", '.'+wsi_extension)
    #slideFolder = '_'.join(slideFileName.split("_")[0:3])
    #slideFolder = slideFileName.split(" - ")[0]
    slideFolder = slideFileName.split('.'+wsi_extension)[0]
    #print(slideFolder)
    #quit()
    heSlide = pv.Image.new_from_file(
        os.path.join(slidesRootFolder, slideFileName), level=magnificationLevel)
    try:
        os.makedirs(os.path.join(tilesRootFolder, slideFolder), exist_ok=False)
        print("Directory ", os.path.join(tilesRootFolder, slideFolder), " Created ")
    except FileExistsError:
        print("Directory ", os.path.join(tilesRootFolder, slideFolder), " already exists")
        continue
    for key in tileDictionary:
        try:
            os.mkdir(os.path.join(tilesRootFolder, slideFolder, key))
            print("Directory ", os.path.join(tilesRootFolder, slideFolder, key), " Created ")
        except FileExistsError:
            print("Directory ", os.path.join(tilesRootFolder, slideFolder, key), " already exists")
        # Iterate over all tiles, extract them and save
        for tile in tqdm(tileDictionary[key]):
            if not os.path.exists(os.path.join(tilesRootFolder, slideFolder, key, slideFolder + '_' + str(tile['x']) + '_' + str(tile['y']) + '_' + str(tile['tileSize']) + '.jpg')):
                #print("tile X:", tile['x'])
                #print("tile Y:", tile['y'])
                try:
                    areaHe = heSlide.extract_area(
                        tile['x'], tile['y'], tile['tileSize'], tile['tileSize'])

                    tile_size = tile['tileSize']

                    # get statistics to compute tile means and standard deviations for normalization during training
                    tile_counter = tile_counter + 1

                    # get tile as numpy ndarray
                    nparea = np.ndarray(buffer=areaHe.write_to_memory(), dtype=format_to_dtype[areaHe.format], shape=[areaHe.height, areaHe.width, areaHe.bands])[...,:3] # remove transparency channel

                    #nparea = self.getTile(tl, writeToNumpy=True)[...,:3] # remove transparency channel
                    nparea = normalize_to_1max(nparea).numpy() # normalize values from 0-255 to 0-1
                    local_channel_sums = np.sum(nparea, axis=(1,2))
                    local_channel_squared_sums = np.sum(np.square(nparea), axis=(1,2))
                    channel_sums = np.add(channel_sums, local_channel_sums)
                    channel_squared_sums = np.add(channel_squared_sums, local_channel_squared_sums)


                except:
                    print('Skipping tile that goes beyond the edge of the WSI...')
                    continue

                areaHe.write_to_file(os.path.join(tilesRootFolder, slideFolder, key, slideFolder + '_' + str(
                    tile['x']) + '_' + str(tile['y']) + '_' + str(tile['tileSize']) + '.jpg'), Q=100)

# save channel means and stds
total_pixels_per_channel = tile_counter * tile_size * tile_size
global_channel_means = np.divide(channel_sums, total_pixels_per_channel)
global_channel_squared_means = np.divide(channel_squared_sums, total_pixels_per_channel)
global_channel_variances = np.subtract(global_channel_squared_means, np.square(global_channel_means))
global_channel_stds = np.sqrt(global_channel_variances * (total_pixels_per_channel / (total_pixels_per_channel-1)))
means_and_stds = {'channel_means': global_channel_means.tolist(), 'channel_stds': global_channel_stds.tolist(), 'tile_count': tile_counter}
print(means_and_stds)
pickle.dump(means_and_stds, open(os.path.join(tilesRootFolder, 'channel_means_and_stds.p'), 'wb'))
