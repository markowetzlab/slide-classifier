import os

import pandas as pd
from slidl.slide import Slide
import geojson

tile_size = 392
tile_overlap = 0.0
model_state_dict_path = 'models/deep-tissue-detector_densenet_state-dict.pt'
tissue_threshold = 0.1

with open('data/tff3_delta_files.txt') as f:
    wsis = [im.rstrip() for im in f]

slide_path = '/media/prew01/DELTA/slides/tff3'

csv_columns = ['image', 'x','y','pixel_x','pixel_y']
tissue_patches = []

images = 0

for image in wsis:
    images += 1
    case_path = os.path.join(slide_path, image)
    output = os.path.join('data/delta_tissue/tff3', image)
    print(f'\rCase {images}/{len(wsis)} {image} processing: ')

    json_annotations = {"type": "FeatureCollection", "features":[]}

    try:
        if os.path.isfile(output + '.pml'):
            slidl_slide = Slide(output + '.pml')
        else:
            slidl_slide = Slide(case_path, level=0).setTileProperties(tileSize=tile_size,tileOverlap=tile_overlap)
            slidl_slide.detectTissue(numWorkers=8, modelStateDictPath=model_state_dict_path)
            # slidl_slide.detectForeground(threshold='triangle')
            slidl_slide.save(fileName=image, folder='data/delta_tissue/tff3')

        
        for tile_address, tile_entry in slidl_slide.tileDictionary.items():
            if tile_entry['tissueLevel'] >= tissue_threshold:
                tissue_patches.append([image, tile_address[0], tile_address[1], tile_entry['x'], tile_entry['y']])
        #         json_annotations['features'].append({
        #             "type": "Feature",
        #             "id": "PathDetectionObject",
        #             "geometry": {
        #             "type": "Polygon",
        #             "coordinates": [
        #                     [
        #                         [tile_entry['x'], tile_entry['y']],
        #                         [tile_entry['x']+ tile_entry['width'], tile_entry['y']],
        #                         [tile_entry['x']+ tile_entry['width'], tile_entry['y'] + tile_entry['width']],
        #                         [tile_entry['x'], tile_entry['y'] + tile_entry['width']],		
        #                         [tile_entry['x'], tile_entry['y']]
        #                     ]	
        #                 ]
        #             },
        #         })
        
        # with open(output+'.geojson', 'w') as annotation_file:
        #     geojson.dump(json_annotations, annotation_file, indent=0)

    except:
        print(f'Image not found {image}')

df = pd.DataFrame(columns=csv_columns, data=tissue_patches)
df.to_csv('tff3_delta_tissue_patches.csv')
print(len(df))