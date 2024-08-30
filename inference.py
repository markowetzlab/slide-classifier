# This file runs tile-level inference on the training, calibration and internal validation cohort
import argparse
import os
import shutil
import time
import warnings

import geojson
import h5py
import monai.transforms as mt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xmltodict
from monai.data import (CSVDataset, DataLoader, OpenSlideWSIReader,
                        PatchWSIDataset)
from tqdm import tqdm

from torchvision import models

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on slides.')

    parser.add_argument("--save_dir", default='results', help="path to folder where inference will be stored")

    #dataset processing
    parser.add_argument("--stain", choices=['he', 'qc', 'p53', 'tff3'], help="Stain to consider H&E (he), quality control (qc) or P53 (p53), TFF3(tff3)")
    parser.add_argument("--process_list", default=None, help="file containing slide-level ground truth to use.")
    parser.add_argument("--slide_path", default='slides', help="slides root folder")
    parser.add_argument("--reader", default='openslide', help="monai slide backend reader ('openslide' or 'cuCIM')")

    #model path and parameters
    parser.add_argument("--model_path", required=True, help="path to stored model weights")

    #preprocessed patch locations and parameters
    parser.add_argument("--patch_path", default='patches', help="path to stored (.h5 or .csv) patch files")
    parser.add_argument("--input_size", default=None, type=int, help="size of tiles to extract")

    #data processing
    parser.add_argument("--batch_size", default=None, help="Batch size. Default is to use values set for architecture to run on 1 GPU.", type=int)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to call for DataLoader")
    
    #thresholds
    parser.add_argument("--he_threshold", default=0.99993, type=float, help="A threshold for detecting gastric cardia in H&E")
    parser.add_argument("--p53_threshold", default= 0.99, type=float, help="A threshold for Goblet cell detection in p53")
    parser.add_argument("--qc_threshold", default=0.99, help="A threshold for detecting gastric cardia in H&E")
    parser.add_argument("--tff3_threshold", default= 0.93, help="A threshold for Goblet cell detection in tff3")
    parser.add_argument("--lcp_cutoff", default=None, help='number of tiles to be considered low confidence positive')
    parser.add_argument("--hcp_cutoff", default=None, help='number of tiles to be considered high confidence positive')
    parser.add_argument("--impute", action='store_true', help="Assume missing data as negative")

    #class variables
    parser.add_argument("--ranked_class", default=None, help='particular class to rank tiles by')
    parser.add_argument("--dysplasia_separate", action='store_false', help="Flag whether to separate the atypia of uncertain significance and dysplasia classes")
    parser.add_argument("--respiratory_separate", action='store_false', help="Flag whether to separate the respiratory mucosa cilia and respiratory mucosa classes")
    parser.add_argument("--gastric_separate", action='store_false', help="Flag whether to separate the tickled up columnar and gastric cardia classes")
    parser.add_argument("--atypia_separate", action='store_false', help="Flag whether to perform the following class split: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other")
    parser.add_argument("--p53_separate", action='store_false', help="Flag whether to perform the following class split: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar")
    
    #outputs
    parser.add_argument("--xml", action='store_true', help='produce annotation files for ASAP in .xml format')
    parser.add_argument("--ndpa", action='store_true', help='produce annotation files for NDP Viewer in .ndpa format')
    parser.add_argument("--json", action='store_true', help='produce annotation files for QuPath in .geoJSON format')
    parser.add_argument("--images", action='store_true', help='save images of positive tiles as png files')

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
    
    model_ver = os.path.splitext(os.path.basename(args.model_path))[0]
    reader = args.reader

    input_directories = {'slide_dir': args.slide_path,
                    'mask_dir': os.path.join(args.patch_path, 'masks'),
                    'patch_dir': os.path.join(args.patch_path, 'patches'),
                    'save_dir': args.save_dir,
                   }
    
    print(f"Outputting inference to: {input_directories['save_dir']}")

    if args.stain == 'he':
        file_name = 'H&E'
        gt_label = 'Atypia'
        classes = ['artifact', 'atypia', 'background', 'gastric_cardia', 'immune_cells', 'intestinal_metaplasia', 'respiratory_mucosa', 'squamous_mucosa']
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
        trained_model = models.vit_l_16(pretrained=False)
        trained_model.heads[-1] = nn.Linear(1024, len(classes))
        trained_model = torch.load(args.model_path)

    elif args.stain == 'qc':
        file_name = 'H&E'
        gt_label = 'QC Report'
        classes = ['Background', 'Gastric-type columnar epithelium', 'Intestinal Metaplasia', 'Respiratory-type columnar epithelium']
        mapping = {'Adequate for pathological review': 1, 'Scant columnar cells': 1, 'Squamous cells only': 0, 'Insufficient cellular material': 0, 'Food material': 0}
        if args.ranked_class is not None:
            ranked_class = args.ranked_class
        else:
            ranked_class = 'Gastric-type columnar epithelium'
        thresh = args.qc_threshold
        if args.lcp_cutoff is not None:
            lcp_threshold = args.lcp_cutoff
        else:
            lcp_threshold = 0
        if args.hcp_cutoff is not None:
            hcp_threshold = args.hcp_cutoff
        else:
            hcp_threshold = 95
        
        trained_model = models.vgg16(pretrained=False)
        trained_model.classifier[6] = nn.Linear(4096, len(classes))
        trained_model.load_state_dict(torch.load(args.model_path).module.state_dict())
        
    elif args.stain == 'p53':
        file_name = 'P53'
        gt_label = 'P53 positive'
        classes = ['aberrant_positive_columnar', 'artifact', 'background', 'immune_cells', 'squamous_mucosa', 'wild_type_columnar']
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
        
        trained_model = models.convnext_large(pretrained=False)
        trained_model.classifier[1] = nn.Linear(1536, len(classes))
        trained_model = torch.load(args.model_path)

    elif args.stain == 'tff3':
        file_name = 'TFF3'
        gt_label = 'TFF3 positive'
        classes = ['Equivocal', 'Negative', 'Positive']
        ranked_class = 'Positive'
        thresh = args.tff3_threshold
        mapping = {'Y': 1, 'N': 0}
        if args.lcp_cutoff is not None:
            lcp_threshold = args.lcp_cutoff
        else:
            lcp_threshold = 3
        if args.hcp_cutoff is not None:
            hcp_threshold = args.hcp_cutoff
        else:
            hcp_threshold = 40
        
        trained_model = models.vgg16(pretrained=False)
        trained_model.classifier[6] = nn.Linear(4096, len(classes))
        trained_model.load_state_dict(torch.load(args.model_path).module.state_dict())
        
    else:
        raise AssertionError(f'stain type must be he/qc, tff3, or p53 but received {str(args.stain)}.')

    # Modify the model to use the updated GELU activation function in later PyTorch versions 
    for name, module in trained_model.named_modules():
        if isinstance(module, nn.GELU):
            exec('trained_model.'+torchmodify(name)+'=nn.GELU()')

    if args.stain == 'he' or args.stain =='p53':
        channel_means = [0.7747305964175918, 0.7421753839460998, 0.7307385516144509]
        channel_stds = [0.2105364799974944, 0.2123423033814637, 0.20617556948731974]
    elif args.stain == 'tff3' or args.stain == 'qc':
        channel_means = [0.485, 0.456, 0.406]
        channel_stds = [0.229, 0.224, 0.225]
    else:
        raise AssertionError(f'stain type must be he/qc, tff3, or p53 but received {str(args.stain)}.')
    print('Channel Means: ', channel_means, '\nChannel Stds: ', channel_stds)
    
    # Use manual batch size if one has been specified
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = 32
    
    if args.input_size is not None:
        input_size = args.patch_size
    else:
        input_size = 224
    
    if torch.cuda.is_available() and (torch.version.hip or torch.version.cuda):
        if torch.cuda.device_count() > 1:
            trained_model = torch.nn.DataParallel(trained_model, device_ids=list(range(torch.cuda.device_count())))
            device = torch.device("cuda")
        else:
            device = torch.device("cuda:0")
            trained_model.to(device)
    else:
        device = torch.device("cpu")
    trained_model.eval()

    eval_transforms = mt.Compose(
            [
                mt.ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=0.0, b_max=1.0),
                mt.ToTensord(keys=("image")),
                mt.TorchVisiond(keys=("image"), name="Normalize", mean=channel_means, std=channel_stds),
                mt.TorchVisiond(keys=("image"), name="Resize", size=(input_size, input_size)),
                mt.ToMetaTensord(keys=("image")),
            ]
        )

    if args.process_list is not None:
        # take in a txt or csv file with slide names to process
        if args.process_list.endswith('.txt'):
            with open(args.process_list, 'r') as file:
                process_list = [line.strip() for line in file.readlines()]
                process_list = pd.DataFrame(process_list, columns=['slide_id'])
        elif args.process_list.endswith('.csv'):
            process_list = pd.read_csv(args.process_list)
        else:
            raise ValueError("Invalid file format. Only txt and csv files are supported.")
    else:
        process_list = []
        for file in os.listdir(input_directories['slide_dir']):
            if file.endswith(('.ndpi','.svs')):
                process_list.append(file)
        slides = sorted(process_list)
        process_list = pd.DataFrame(slides, columns=['slide_id'])

    # Extract the sample ID and pot ID from the slide name
    sample_id = process_list['slide_id'].str.split(' ')
    process_list['CYT ID'] = sample_id.str[0]
    process_list['Pot ID'] = sample_id.str[2]
    process_list.sort_index(inplace=True)

    records = []

    for index, row in process_list.iterrows():
        case = row["CYT ID"]
        slide = row['slide_id']
        # get base string of slide name
        slide_stem = os.path.splitext(os.path.basename(slide))[0]
        try:
            wsi_path = os.path.join(input_directories['slide_dir'], slide)
        except:
            print(f'File {slide} not found.')
            continue

        if not os.path.exists(wsi_path):
            print(f'File {wsi_path} not found.')
            continue
        print(f'\rProcessing case {index+1}/{len(process_list)} {case}: ', end='')

        output_dir = os.path.join(args.save_dir, slide_stem)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        mask_file = os.path.join(input_directories['mask_dir'], slide_stem+'.jpg')
        patch_file = os.path.join(input_directories['patch_dir'], slide_stem+'.h5')
        patch_file = h5py.File(patch_file)['coords']
        patch_size = patch_file.attrs['patch_size']
        patch_level = patch_file.attrs['patch_level']

        tile_inference = os.path.join(output_dir, model_ver+'.csv')

        if os.path.isfile(tile_inference):
            print(f'Inference for {case} already exists.')
            predictions = pd.read_csv(tile_inference)
        else:
            # copy mask file to output directory for reference
            destination_file = mask_file.replace(slide_stem, 'mask')
            destination_file = destination_file.replace(input_directories['mask_dir'], output_dir)
            shutil.copy2(mask_file, destination_file)
            print(f'File copied from {mask_file} to {destination_file}')
            
            locations = pd.DataFrame(np.array(patch_file), columns=['x_min','y_min'])
            locations['image'] = wsi_path
            print('Number of tiles:',len(locations))
            #monai coordinates are trasposed
            patch_locations = CSVDataset(locations,
                                col_groups={"image": "image", "location": ["y_min", "x_min"]},
                            )

            dataset = PatchWSIDataset(
                data=patch_locations,
                patch_size=patch_size,
                patch_level=patch_level,
                include_label=False,
                center_location=False,
                transform = eval_transforms,
                reader = reader
            )

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            tile_predictions = []

            since = time.time()

            with torch.no_grad():
                for inputs in tqdm(dataloader, disable=args.silent):
                    tile = inputs['image'].to(device)
                    tile_location = inputs['image'].meta['location'].numpy()
                    
                    output = trained_model(tile)

                    batch_prediction = torch.nn.functional.softmax(output, dim=1).cpu().data.numpy()
                    predictions = np.concatenate((tile_location, batch_prediction), axis=1)
                    for i in range(len(predictions)):
                        tile_predictions.append(predictions[i])

            columns = ['y_min', 'x_min'] + classes
            predictions = pd.DataFrame(tile_predictions, columns=columns)
            predictions['x_min'] = predictions['x_min'].astype(int)
            predictions['y_min'] = predictions['y_min'].astype(int)
            predictions['x_max'] = predictions['x_min'] + patch_size
            predictions['y_max'] = predictions['y_min'] + patch_size
            predictions = predictions.reindex(columns=['x_min', 'y_min', 'x_max', 'y_max'] + classes)
            predictions.to_csv(tile_inference, index=False)

            time_elapsed = time.time() - since
            print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        positive_tiles = (predictions[ranked_class] > thresh).sum()
        if positive_tiles >= hcp_threshold:
            algorithm_result = 3
        elif (positive_tiles < hcp_threshold) & (positive_tiles > lcp_threshold):
            algorithm_result = 2
        else:
            algorithm_result = 1

        annotation_path = None
        positive = predictions[predictions[ranked_class] > thresh]
        positive = positive.sort_values(by=ranked_class, ascending=False)

        if len(positive) == 0:
            print(f'No annotations found at current threshold for {slide_stem}')
        else:
            if args.xml:
                annotation_path = os.path.join(output_dir, model_ver+'.xml')
                if not os.path.exists(annotation_path):
                    # Make ASAP file
                    xml_header = """<?xml version="1.0"?><ASAP_Annotations>\t<Annotations>\n"""
                    xml_tail =  f"""\t</Annotations>\t<AnnotationGroups>\t\t<Group Name="{ranked_class}" PartOfGroup="None" Color="#64FE2E">\t\t\t<Attributes />\t\t</Group>\t</AnnotationGroups></ASAP_Annotations>\n"""

                    xml_annotations = ""
                    for index, tile in positive.iterrows():
                        xml_annotations += ("\t\t<Annotation Name=\""+str(ranked_class + '_' + str(tile[ranked_class]))+"\" Type=\"Polygon\" PartOfGroup=\""+ranked_class+"\" Color=\"#F4FA58\">\n" +
                                            "\t\t\t<Coordinates>\n" +
                                            "\t\t\t\t<Coordinate Order=\"0\" X=\""+str(tile['x_min'])+"\" Y=\""+str(tile['y_min'])+"\" />\n" +
                                            "\t\t\t\t<Coordinate Order=\"1\" X=\""+str(tile['x_max'])+"\" Y=\""+str(tile['y_min'])+"\" />\n" +
                                            "\t\t\t\t<Coordinate Order=\"2\" X=\""+str(tile['x_max'])+"\" Y=\""+str(tile['y_max'])+"\" />\n" +
                                            "\t\t\t\t<Coordinate Order=\"3\" X=\""+str(tile['x_min'])+"\" Y=\""+str(tile['y_max'])+"\" />\n" +
                                            "\t\t\t</Coordinates>\n" +
                                            "\t\t</Annotation>\n"
                                            )
                    print('Creating automated annotation file for '+ slide_stem)
                    with open(annotation_path, "w") as f:
                        f.write(xml_header + xml_annotations + xml_tail)
                else:
                    print(f'Automated xml annotation file for {slide_stem} already exists.')

            if args.ndpa:
                annotation_path = os.path.join(output_dir, slide +'.ndpa')

                if not os.path.exists(annotation_path):
                    reader = OpenSlideWSIReader(level=patch_level)
                    if not slide.endswith('.ndpi'):
                        raise AssertionError('Only .ndpi files are supported for NDP Viewer annotations.')
                    slide = reader.read(wsi_path)

                    #convert center of slide to nanometers (*1000)
                    conversion_factor_x = float(slide.properties.get('openslide.mpp-x'))*1000
                    conversion_factor_y = float(slide.properties.get('openslide.mpp-y'))*1000

                    width_nm = slide.dimensions[0]/2 * conversion_factor_x
                    height_nm = slide.dimensions[1]/2 * conversion_factor_y
                    
                    x_offset = int(width_nm) - int(slide.properties.get('hamamatsu.XOffsetFromSlideCentre'))
                    y_offset = int(height_nm) - int(slide.properties.get('hamamatsu.YOffsetFromSlideCentre'))
                    
                    ndp_view_list = []
                    for index, tile in positive.iterrows():
                        coordinates = [
                                        [tile['x_min'], tile['y_min']],
                                        [tile['x_max'], tile['y_min']],
                                        [tile['x_max'], tile['y_max']],
                                        [tile['x_min'], tile['y_max']],        
                                    ]   

                        # Calculate x_mean and y_mean from pixel to nm
                        x_mean = int(sum([coord[0] * conversion_factor_x for coord in coordinates]) / len(coordinates))
                        y_mean = int(sum([coord[1] * conversion_factor_y for coord in coordinates]) / len(coordinates))
                        
                        ndp_coords = []
                        # Convert the coordinates to ndp
                        for coord in coordinates:
                            ndp_coords.append({'x': int((coord[0] * conversion_factor_x) - x_offset), 'y': int((coord[1] * conversion_factor_y) - y_offset)})
                        
                        ndp_view_list.append({
                            '@id': str(index+1),
                            'title': str(ranked_class),
                            'details': 'These are the details of Annotation '+str(index+1),
                            'coordformat': 'nanometers',
                            'lens': '3.628447',
                            'x': str(x_mean),
                            'y': str(y_mean),
                            'z': '0',
                            'showtitle': '1',
                            'showhistogram': '0',
                            'showlineprofile': '0',
                            'annotation': {
                                '@type': 'freehand',
                                '@displayname': 'AnnotateRectangle',
                                '@color': '#00ff00',
                                'measuretype': '0',
                                'closed': '1',
                                'pointlist': {'point': ndp_coords},
                                'specialtype': 'rectangle'
                            }
                        }
                        )
                    
                    # Make the xml
                    xml = {'annotations': {'ndpviewstate': ndp_view_list}}

                    print('Creating automated ndpa annotation file for '+ slide_stem)
                    with open(annotation_path, "w") as f:
                        f.write(xmltodict.unparse(xml, pretty=True))
                else:
                    print(f'Automated ndpa annotation file for {annotation_path} already exists.')

            if args.json:
                annotation_path = os.path.join(output_dir, model_ver+'.geojson')
                if not os.path.exists(annotation_path):
                    json_annotations = {"type": "FeatureCollection", "features":[]}
                    for index, tile in positive.iterrows():
                        color = [0, 0, 255]
                        status = str(ranked_class)

                        json_annotations['features'].append({
                            "type": "Feature",
                            "id": "PathDetectionObject",
                            "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                    [
                                        [tile['x_min'], tile['y_min']],
                                        [tile['x_max'], tile['y_min']],
                                        [tile['x_max'], tile['y_max']],
                                        [tile['x_min'], tile['y_max']],        
                                        [tile['x_min'], tile['y_min']]
                                    ]   
                                ]
                            },
                            "properties": {
                                "objectType": "annotation",
                                "name": str(status)+'_'+str(round(tile[ranked_class], 4))+'_'+str(tile['x_min']) +'_'+str(tile['y_min']),
                                "classification": {
                                    "name": status,
                                    "color": color
                                }
                            }
                        })
                    print('Creating automated geojson annotation file for ' + slide_stem)
                    with open(annotation_path, "w") as f:
                        geojson.dump(json_annotations, f, indent=0)
                else:
                    print(f'Automated geojson annotation file for {slide_stem} already exists')

            if args.images:
                if len(positive) == 0:
                    print(f'\rNo annotations found above current threshold for {slide_stem}')
                else:
                    reader = OpenSlideWSIReader(level=patch_level)
                    slide = reader.read(wsi_path)
                    slide_images = os.path.join(output_dir, 'images')
                    print(f'Creating images for {slide_stem} at {slide_images}')
                    if not os.path.exists(slide_images):
                        os.makedirs(slide_images, exist_ok=True)
                    tiles = 0
                    for index, tile in positive.iterrows():
                        tiles += 1
                        
                        status = str(ranked_class)

                        x_min, y_min = int(tile['x_min']), int(tile['y_min'])
                        w, h = int(tile['x_max']) - x_min, int(tile['y_max']) - y_min

                        zoom_out_factor = 2  # Define how much to zoom out. For example, 2 means doubling the width and height.
                        # Adjust the top-left corner to keep the initial ROI centered
                        new_x = x_min - (w * (zoom_out_factor - 1) // 2)
                        new_y = y_min - (h * (zoom_out_factor - 1) // 2)
                        # Adjust the width and height
                        new_w = w * zoom_out_factor
                        new_h = h * zoom_out_factor

                        tile_image = slide.read_region((new_x, new_y), patch_level, (new_w, new_h))
                        tile_image = tile_image.convert('RGB')
                        tile_image.save(os.path.join(slide_images, f'{status}_{str(round(tile[ranked_class], 4))}_{str(int(tile["x_min"]))}_{str(int(tile["y_min"]))}.png'))
                        #produce the top 5 tiles that would otherwise be classed as negative for reference
                        if tiles == 5:
                            break
                
        record = {
            'algorithm_cyted_sample_id': case,
            'slide_filename': slide_stem,
            'positive_tiles': positive_tiles,
            'algorithm_result': algorithm_result,
            'tile_mapping': os.path.basename(annotation_path) if annotation_path else None,
            'algorithm_version': model_ver,
        }
        records.append(record)

    df = pd.DataFrame.from_dict(records)
    date = time.strftime('%d%m%y')
    df.to_csv(os.path.join(args.save_dir, args.stain+f'_process_list_{date}.csv'), index=False)
    print(f'Number of HCP slides: {(df["algorithm_result"] == 3).sum()}')
    print(f'Number of LCP slides: {(df["algorithm_result"] == 2).sum()}')
    print(f'Number of Negative slides: {(df["algorithm_result"] == 1).sum()}')

#TODO implement MONAI version of Metrics Reloaded
#https://github.com/Project-MONAI/MetricsReloaded