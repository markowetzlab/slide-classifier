import os
import argparse
from tqdm import tqdm

from slidl.slide import Slide
import pandas as pd

from dataset_processing import class_parser

def parse_args():
	parser = argparse.ArgumentParser(description='Run inference on slides.')

	#dataset processing
	parser.add_argument("--stain", choices=['he', 'p53'], required=True, help="he or p53")
	parser.add_argument("--labels", help="file containing slide-level ground truth to use.'")

	#slide paths and tile properties
	parser.add_argument("--slide_path", required=True, help="slides root folder")
	parser.add_argument("--xml_path", required=True, help="slides root folder")
	parser.add_argument("--format", default=".ndpi", help="extension of whole slide image")
	parser.add_argument("--tile_size", default=400, help="architecture tile size")
	parser.add_argument("--overlap", default=0, help="what fraction of the tile edge neighboring tiles should overlap horizontally and vertically (default is 0)")
	parser.add_argument("--foreground_only", action='store_true', help="Foreground with tissue only")
	
	#class variables
	parser.add_argument("--dysplasia_separate", action='store_false', help="Flag whether to separate the atypia of uncertain significance and dysplasia classes")
	parser.add_argument("--respiratory_separate", action='store_false', help="Flag whether to separate the respiratory mucosa cilia and respiratory mucosa classes")
	parser.add_argument("--gastric_separate", action='store_false', help="Flag whether to separate the tickled up columnar and gastric cardia classes")
	parser.add_argument("--atypia_separate", action='store_false', help="Flag whether to perform the following class split: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other")
	parser.add_argument("--p53_separate", action='store_false', help="Flag whether to perform the following class split: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar")
	
	parser.add_argument('--silent', action='store_true', help='Flag which silences terminal outputs')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()

	slide_path = args.slide_path
	xml_path = args.xml_path
	tile_size = args.tile_size
	overlap = args.overlap
	output_path = args.output

	list_of_slides = []
	list_of_annotations = []
	classes_to_extract = class_parser(args.stain, args.dysplasia_separate, args.respiratory_separate, args.gastric_separate, args.atypia_separate, args.p53_separate)

	data = {}

	for case_file, xml_file in zip(list_of_slides, list_of_annotations):
		print(f'Processing case {case_file} and {xml_file}')
		
		case_path = os.path.join(slide_path, case_file)
		case_xml = os.path.join(xml_path, xml_file)
		
		slidl_slide = Slide(case_path)
		slidl_slide.setTileProperties(tileSize=tile_size, tileOverlap=float(overlap))
		slidl_slide = slidl_slide.addAnnotations(case_xml)

		for tile_address, tile_entry in tqdm(slidl_slide.tileDictionary.items(), disable=args.silent):
			row = []
			row.append(case_file)
			row.append(tile_address['x'])
			row.append(tile_address['y'])
			for classes in classes_to_extract:
				if classes in tile_entry:
					label = 1
				else:
					label = 0
				row.append(label)
			data.update(row)
	
	df = pd.DataFrame.from_dict(data)
	print(f'Num of tiles: {len(df)}')
	df.to_csv(output_path+'.csv', index=False)
