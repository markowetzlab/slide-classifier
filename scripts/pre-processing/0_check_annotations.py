# 0_checkAnnotations.py
#
# Adam Berman and Marcel Gehrung
#
# This file checks the integrity of annotations on all .xml files which contain ROIs for H&E and P53 slides
# It creates so-called tile-dictionaries
#

import argparse
import glob, os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from shapely import geometry
import pyvips as pv
import numpy as np
import math
from tqdm import tqdm
import pickle, sys
sys.path.append('/home/cri.camres.org/berman01/pathml') # install from here: https://github.com/9xg/pathml
from pathml import slide

parser = argparse.ArgumentParser(description='Create tile dictionaries from annotation ROIs.')
parser.add_argument("--stain", required=True, help="he or p53")
parser.add_argument("--slidesrootfolder", required=True, help="slides root folder")
parser.add_argument("--annotationsrootfolder", required=True, help="annotations root folder containing xml ASAP annotation files")
parser.add_argument("--output", required=True, help="folder to store newly created tile dictionary folder in")
parser.add_argument("--combineatypia", default="True", help="True or False as to whether to combine the atypia of uncertain significance and dysplasia classes (default is True)")
parser.add_argument("--combinerespiratory", default="True", help="True or False as to whether to combine the respiratory mucosa cilia and respiratory mucosa classes (default is True)")
parser.add_argument("--combinegastriccardia", default="True", help="True or False as to whether to combine the tickled up columnar and gastric cardia classes (default is True)")
parser.add_argument("--useatypiaclasscombiners", default="True", help="True or False as to whether to perform the following class mergers: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other (default is True)")
parser.add_argument("--usep53classcombiners", default="True", help="True or False as to whether to perform the following class mergers: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar (default is True)")
parser.add_argument("--wsiextension", default="ndpi", help="extension of whole slide image without full stop")
args = parser.parse_args()
print(args)

whichStain = args.stain # he or p53
if whichStain not in ['he', 'p53']:
    raise Warning("-stain argument must be either 'he' or 'p53'")
tileSize=400 # extract 224 by 224 pixel tiles at given magnification level, previously we used 400 at 40x
magnificationLevel = 0 # 1 = 20(x) or 0 = 40(x)
enforceMagnificationLock = True # Only permit slides which were scanned at around 0.25 per micron
annotationCoverageThreshold = {'he': 0.33, 'p53': 0.33}#'tff3':0.66}
# If the scan was successfully checked to be at 40x, all magnification coordinates have to the scaled by the downsample factor of the level difference (40x vs 20x)
magnificationLevelConverted = 40 if magnificationLevel==0 else 20
wsi_extension = args.wsiextension

outputFolderName = 'tileDictionaries-'+whichStain+'-'+str(tileSize)+'px-at-'+str(magnificationLevelConverted)+'x'
outputFolderPath = os.path.join(args.output, outputFolderName)

if not os.path.exists(outputFolderPath):
    os.makedirs(outputFolderPath)

verbose = True
slidesRootFolder = args.slidesrootfolder
heAnnotationsFileList = glob.glob(os.path.join(args.annotationsrootfolder, "BEST*.xml")) # path to best2 annotations

if whichStain == "he":
    if args.useatypiaclasscombiners in ["True", "true", "TRUE"]:
        classNames = ['artifact', 'atypia', 'background', 'gastric_cardia', 'immune_cells',
                        'intestinal_metaplasia', 'respiratory_mucosa', 'squamous_mucosa']

    else:

        if args.combineatypia in ["True", "true", "TRUE"]:
            if args.combinerespiratory in ["True", "true", "TRUE"]:
                if args.combinegastriccardia in ["True", "true", "TRUE"]:
                    classNames = ['artifact', 'atypia', 'background', 'gastric_cardia', 'immune_cells',
                                    'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'squamous_mucosa']
                else:
                    classNames = ['artifact', 'atypia', 'background', 'gastric_cardia', 'immune_cells',
                                    'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'squamous_mucosa', 'tickled_up_columnar']
            else:
                if args.combinegastriccardia in ["True", "true", "TRUE"]:
                    classNames = ['artifact', 'atypia', 'background', 'gastric_cardia', 'immune_cells',
                                    'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'respiratory_mucosa_cilia', 'squamous_mucosa']
                else:
                    classNames = ['artifact', 'atypia', 'background', 'gastric_cardia', 'immune_cells',
                                    'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'respiratory_mucosa_cilia', 'squamous_mucosa', 'tickled_up_columnar']

        else:
            if args.combinerespiratory in ["True", "true", "TRUE"]:
                if args.combinegastriccardia in ["True", "true", "TRUE"]:
                    classNames = ['artifact', 'atypia_of_uncertain_significance', 'background', 'dysplasia', 'gastric_cardia',
                                    'immune_cells', 'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'squamous_mucosa']
                else:
                    classNames = ['artifact', 'atypia_of_uncertain_significance', 'background', 'dysplasia', 'gastric_cardia',
                                    'immune_cells', 'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'squamous_mucosa', 'tickled_up_columnar']
            else:
                if args.combinegastriccardia in ["True", "true", "TRUE"]:
                    classNames = ['artifact', 'atypia_of_uncertain_significance', 'background', 'dysplasia', 'gastric_cardia',
                                    'immune_cells', 'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'respiratory_mucosa_cilia', 'squamous_mucosa']
                else:
                    classNames = ['artifact', 'atypia_of_uncertain_significance', 'background', 'dysplasia', 'gastric_cardia',
                                    'immune_cells', 'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'respiratory_mucosa_cilia', 'squamous_mucosa', 'tickled_up_columnar']
elif whichStain == "p53":
    if args.usep53classcombiners:
        classNames = ['aberrant_positive_columnar', 'artifact', 'background', 'immune_cells', 'squamous_mucosa', 'wild_type_columnar']
    else:
        classNames = ['aberrant_positive_columnar', 'artifact', 'background', 'equivocal_columnar',
                        'immune_cells', 'nonspecific_background', 'oral_bacteria', 'respiratory_mucosa', 'squamous_mucosa', 'wild_type_columnar']
    #class_names = ['aberrant_negative_columnar', 'aberrant_positive_columnar', 'artifact', 'background', 'gastric_cardia',
    #                'immune_cells', 'other', 'respiratory_mucosa', 'squamous_mucosa', 'unspecific_stain', 'wild_type_columnar']

#if whichStain is 'he':
#    classNames = ['squamous_mucosa', 'gastric_cardia', 'intestinal_metaplasia', 'atypia_of_uncertain_significance', 'dysplasia', 'other', 'artifact', 'respiratory_mucosa', 'immune_cells', 'background'] # Specify class names which at the same time are the names of the respective directory
#elif whichStain is 'tff3':
#    classNames = ['Equivocal', 'Positive', 'Negative','Stain']

# Cases with pathologist-validated ground truth atypia annotations
heCasesToUse = ['BEST2 CAM 0001', 'BEST2 CAM 0007', 'BEST2 CAM 0018', 'BEST2 CAM 0027', 'BEST2 CAM 0029', 'BEST2 CAM 0053', 'BEST2 CAM 0060', 'BEST2 CAM 0065', 'BEST2 CAM 0078', 'BEST2 CAM 0092',
                'BEST2 CAM 0127', 'BEST2 CAM 0158', 'BEST2 CAM 0250', 'BEST2 CAM 0255', 'BEST2 CAM 0287', 'BEST2 CAM 0315', 'BEST2 CAM 0351', 'BEST2 CAM 0392', 'BEST2 CAM 0412', 'BEST2 CAM 0416',
                'BEST2 CAM 0419', 'BEST2 CAM 0423', 'BEST2 CAM 0430', 'BEST2 CAM 0431', 'BEST2 CAM 0441', 'BEST2 CAM 0444', 'BEST2 CAM 0445', 'BEST2 CAM 0448', 'BEST2 CAM 0467', 'BEST2 CAM 0475',
                'BEST2 CAM 0483', 'BEST2 CAM 0489', 'BEST2 CAM 0491', 'BEST2 CAM 0492', 'BEST2 CAM 0496', 'BEST2 CAM 0506', 'BEST2 CDD 0004', 'BEST2 CDD 0019', 'BEST2 NEW 0004', 'BEST2 NEW 0005',
                'BEST2 NEW 0011', 'BEST2 NEW 0035', 'BEST2 NEW 0061', 'BEST2 NEW 0063', 'BEST2 NEW 0071', 'BEST2 NEW 0079', 'BEST2 NEW 0084', 'BEST2 NEW 0096', 'BEST2 NEW 0097', 'BEST2 NEW 0112',
                'BEST2 NEW 0120', 'BEST2 NEW 0124', 'BEST2 NEW 0127', 'BEST2 NEW 0130', 'BEST2 NEW 0131', 'BEST2 NEW 0147', 'BEST2 NEW 0155', 'BEST2 NEW 0177', 'BEST2 NEW 0178', 'BEST2 NEW 0193',
                'BEST2 NEW 0194', 'BEST2 NEW 0195', 'BEST2 NEW 0199', 'BEST2 NEW 0203', 'BEST2 NEW 0204', 'BEST2 NEW 0205', 'BEST2 NOT 0011', 'BEST2 NOT 0021', 'BEST2 NOT 0027', 'BEST2 NOT 0034',
                'BEST2 NOT 0036', 'BEST2 NOT 0037', 'BEST2 NOT 0047', 'BEST2 NOT 0061', 'BEST2 NOT 0068', 'BEST2 NOT 0070', 'BEST2 NOT 0085', 'BEST2 NOT 0088', 'BEST2 NOT 0090', 'BEST2 NOT 0104',
                'BEST2 NOT 0116', 'BEST2 NOT 0121', 'BEST2 NOT 0123', 'BEST2 NOT 0138', 'BEST2 NOT 0143', 'BEST2 NOT 0144', 'BEST2 NTE 0003', 'BEST2 NTE 0016', 'BEST2 POR 0031', 'BEST2 POR 0032',
                'BEST2 POR 0036', 'BEST2 POR 0039', 'BEST2 POR 0040', 'BEST2 STM 0012', 'BEST2 STM 0016', 'BEST2 STY 0009', 'BEST2 UCL 0001', 'BEST2 UCL 0003', 'BEST2 UCL 0004', 'BEST2 UCL 0022',
                'BEST2 UCL 0028', 'BEST2 UCL 0038', 'BEST2 UCL 0039', 'BEST2 UCL 0041', 'BEST2 UCL 0044', 'BEST2 UCL 0047', 'BEST2 UCL 0055', 'BEST2 UCL 0059', 'BEST2 UCL 0115', 'BEST2 UCL 0122',
                'BEST2 UCL 0136', 'BEST2 UCL 0157', 'BEST2 UCL 0176', 'BEST2 UCL 0178', 'BEST2 UCL 0215', 'BEST2 UCL 0216', 'BEST2 UCL 0223', 'BEST2 UCL 0248', 'BEST2 UCL 0249', 'BEST2 UCL 0252',
                'BEST2 UCL 0253', 'BEST2 UCL 0258', 'BEST2 UCL 0261', 'BEST2 UCL 0268', 'BEST2 UCL 0270', 'BEST2 UCL 0271', 'BEST2 UCL 0272', 'BEST2 UCL 0276', 'BEST2 UCL 0278', 'BEST2 UCL 0284',
                'BEST2 UCL 0286', 'BEST2 UCL 0287', 'BEST2 WEL 0014']
p53CasesToUse = ['BEST2 CAM 0001', 'BEST2 CAM 0011', 'BEST2 CAM 0053', 'BEST2 CAM 0069', 'BEST2 CAM 0081', 'BEST2 CAM 0127', 'BEST2 CAM 0162', 'BEST2 CAM 0250', 'BEST2 CAM 0255', 'BEST2 CAM 0287',
                 'BEST2 CAM 0395', 'BEST2 CAM 0412', 'BEST2 CAM 0416', 'BEST2 CAM 0430', 'BEST2 CAM 0441', 'BEST2 CAM 0444', 'BEST2 CAM 0452', 'BEST2 CAM 0467', 'BEST2 CAM 0472', 'BEST2 CAM 0475',
                 'BEST2 CAM 0483', 'BEST2 CAM 0488', 'BEST2 CAM 0491', 'BEST2 CAM 0496', 'BEST2 CAM 0506', 'BEST2 CDD 0018', 'BEST2 NEW 0005', 'BEST2 NEW 0011', 'BEST2 NEW 0021', 'BEST2 NEW 0038',
                 'BEST2 NEW 0079', 'BEST2 NEW 0096', 'BEST2 NEW 0097', 'BEST2 NEW 0104', 'BEST2 NEW 0124', 'BEST2 NEW 0127', 'BEST2 NEW 0130', 'BEST2 NEW 0131', 'BEST2 NEW 0149', 'BEST2 NEW 0151',
                 'BEST2 NEW 0155', 'BEST2 NEW 0187', 'BEST2 NEW 0189', 'BEST2 NEW 0194', 'BEST2 NEW 0195', 'BEST2 NEW 0199', 'BEST2 NEW 0203', 'BEST2 NEW 0204', 'BEST2 NEW 0205', 'BEST2 NOT 0022',
                 'BEST2 NOT 0027', 'BEST2 NOT 0036', 'BEST2 NOT 0037', 'BEST2 NOT 0047', 'BEST2 NOT 0068', 'BEST2 NOT 0070', 'BEST2 NOT 0085', 'BEST2 NOT 0087', 'BEST2 NOT 0088', 'BEST2 NOT 0104',
                 'BEST2 NOT 0116', 'BEST2 NOT 0124', 'BEST2 NOT 0136', 'BEST2 NOT 0138', 'BEST2 PHH 0009', 'BEST2 POR 0005', 'BEST2 POR 0016', 'BEST2 POR 0025', 'BEST2 POR 0028', 'BEST2 POR 0040',
                 'BEST2 PWH 0003', 'BEST2 STY 0004', 'BEST2 UCL 0003', 'BEST2 UCL 0023', 'BEST2 UCL 0028', 'BEST2 UCL 0031', 'BEST2 UCL 0035', 'BEST2 UCL 0042', 'BEST2 UCL 0043', 'BEST2 UCL 0044',
                 'BEST2 UCL 0047', 'BEST2 UCL 0049', 'BEST2 UCL 0055', 'BEST2 UCL 0069', 'BEST2 UCL 0102', 'BEST2 UCL 0122', 'BEST2 UCL 0151', 'BEST2 UCL 0176', 'BEST2 UCL 0215', 'BEST2 UCL 0250',
                 'BEST2 UCL 0252', 'BEST2 UCL 0258', 'BEST2 UCL 0261', 'BEST2 UCL 0267', 'BEST2 UCL 0268', 'BEST2 UCL 0272', 'BEST2 UCL 0278', 'BEST2 UCL 0287', 'BEST2 WEL 0003', 'BEST2 WEL 0006']
casesToUse = heCasesToUse if whichStain == 'he' else p53CasesToUse

for annotationFile in tqdm(heAnnotationsFileList,desc="Processing slides"):

    #best2Id = os.path.split(annotationFile)[-1].split(' '+whichStain.upper())[0]
    best2Id = ' '.join(os.path.split(annotationFile)[-1].split(' ')[:3])
    if best2Id in casesToUse:
        #continue
        #if whichStain == 'p53' and best2Id == "BEST2 NEW 0011": # current scan of this slide was done at 20x rather than 40x and is therefore being rescanned (rescanned and fixed as of late June 2022)
        #    raise Warning('REPLACE P53 BEST2 NEW 0011 ANNOTATIONS FOR 40x SCANNED SLIDE')
            #continue

        tileDictionary = {listKey:[] for listKey in classNames}
        slideFileName = os.path.split(annotationFile)[-1].replace("xml", wsi_extension)
        #slideFolder = '_'.join(slideFileName.split("_")[0:3])

        print(slideFileName)
        #if os.path.isfile('data/tileDictionaries-tff3-300px-at-40x/'+slideFileName.replace(wsi_extension,'')+'.p'):
        if os.path.isfile(os.path.join(outputFolderPath, slideFileName.replace('.'+wsi_extension,'')+'.p')):
            print("Case already processed. Skipping...")
            continue
        heSlidePML = slide.Slide(os.path.join(slidesRootFolder, slideFileName),level=magnificationLevel)

        # Check whether the slide was scanned uniformly or whether there is any resolution corruption
        #print("X:", round(float(heSlidePML.slideProperties['openslide.mpp-x']), 2))
        #print("Y:", round(float(heSlidePML.slideProperties['openslide.mpp-y']), 2))
        if round(float(heSlidePML.slideProperties['openslide.mpp-x']), 2) != round(float(heSlidePML.slideProperties['openslide.mpp-y']), 2):
            raise Warning('Mismatch between X and Y resolution (microns per pixel)')
        # Check whether the slides was scanned at 40x, otherwise fail
        #print(float(heSlidePML.slideProperties['openslide.mpp-x']))
        if (round(float(heSlidePML.slideProperties['openslide.mpp-x']),2) > 0.3 or round(float(heSlidePML.slideProperties['openslide.mpp-x']),2) < 0.2) and enforceMagnificationLock:
            raise Warning('Slide not scanned at 40x')
        # Calculate the pixel size based on provided tile size and magnification level
        rescaledMicronsPerPixel = float(heSlidePML.slideProperties['openslide.mpp-x'])*float(heSlidePML.slideProperties['openslide.level['+str(magnificationLevel)+'].downsample'])
        #if verbose: print("Given the properties of this scan, the resulting tile size will correspond to "+str(round(tileSize*rescaledMicronsPerPixel,2))+" μm edge length")

        # Calculate scaling factor for annotations
        annotationScalingFactor = float(heSlidePML.slideProperties['openslide.level[0].downsample'])/float(heSlidePML.slideProperties['openslide.level['+str(magnificationLevel)+'].downsample'])

        print("Scale: "+str(annotationScalingFactor))

        # Extract slide dimensions
        slideWidth = int(heSlidePML.slideProperties['width'])
        slideHeight = int(heSlidePML.slideProperties['height'])

        if verbose: print('Opening file: ' + annotationFile)
        tree = ET.parse(annotationFile) # Open .xml file
        root = tree.getroot() # Get root of .xml tree
        if root.tag == "ASAP_Annotations": # Check whether we actually deal with an ASAP .xml file
            if verbose: print('.xml file identified as ASAP annotation collection') # Display number of found annotations
        else:
            raise Warning('Not an ASAP .xml file')
        allHeAnnotations = root.find('Annotations') # Find all annotations for this slide
        if verbose: print('XML file valid - ' + str(len(allHeAnnotations)) + ' annotations found.') # Display number of found annotations

        # Generate a list of tile coordinates which we can extract
        for annotation in tqdm(allHeAnnotations,desc="Parsing annotations"):
            if annotation.attrib['PartOfGroup'] not in ['unsure', 'aberrant_negative_columnar', 'garbage', 'IM_cardia', 'reactive_columnar', 'stromal_fibroblasts', 'candida']: # exclude annotations in the unsure, aberrant_negative_columnar, stromal fibroblasts, candida, and garbage classes
                #print(annotation.attrib['Name'])
                annotationTree = annotation.find('Coordinates')
                x = []
                y = []
                polygon = []
                for coordinate in annotationTree:
                    info = coordinate.attrib
                    polygon.append((float(info['X'])*annotationScalingFactor, float(info['Y'])*annotationScalingFactor))

                polygonNp = np.asarray(polygon)
                polygonNp[:,1] = slideHeight-polygonNp[:,1]
                poly = geometry.Polygon(polygonNp).buffer(0)

                topLeftCorner = (min(polygonNp[:,0]),max(polygonNp[:,1]))
                bottomRightCorner = (max(polygonNp[:,0]),min(polygonNp[:,1]))
                tilesInXax = math.ceil((bottomRightCorner[0] - topLeftCorner[0])/tileSize)
                tilesInYax = math.ceil((topLeftCorner[1] - bottomRightCorner[1])/tileSize)
                x = poly.exterior.coords.xy[0]
                y = poly.exterior.coords.xy[1]

                # The annotation is large enough to extract multiple tiles from it
                if poly.area >1.5*tileSize**2:
                    for xTile in range(tilesInXax):
                        for yTile in range(tilesInYax):
                            minX = topLeftCorner[0]+tileSize*xTile
                            minY = topLeftCorner[1]-tileSize*yTile
                            maxX = topLeftCorner[0]+tileSize*xTile+tileSize
                            maxY = topLeftCorner[1]-tileSize*yTile-tileSize
                            tileBox = geometry.box(minX,minY,maxX,maxY)
                            intersectingArea = poly.intersection(tileBox).area/tileSize**2

                            if intersectingArea > annotationCoverageThreshold[whichStain]:
                                if whichStain == 'he':
                                    # combine atypia_of_uncertain_significance and dysplasia tiles into one class: atypia
                                    if annotation.attrib['PartOfGroup'] in ['atypia_of_uncertain_significance', 'dysplasia']:
                                        if args.useatypiaclasscombiners in ["True", "true", "TRUE"] or args.combineatypia in ["True", "true", "TRUE"]:
                                            tileDictionary['atypia'].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                        else:
                                            tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                    elif annotation.attrib['PartOfGroup'] in ['respiratory_mucosa_cilia']:
                                        if args.useatypiaclasscombiners in ["True", "true", "TRUE"] or args.combinerespiratory in ["True", "true", "TRUE"]:
                                            tileDictionary['respiratory_mucosa'].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                        else:
                                            tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                    elif annotation.attrib['PartOfGroup'] in ['tickled_up_columnar']:
                                        if args.useatypiaclasscombiners in ["True", "true", "TRUE"] or args.combinegastriccardia in ["True", "true", "TRUE"]:
                                            tileDictionary['gastric_cardia'].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                        else:
                                            tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                    elif annotation.attrib['PartOfGroup'] in ['other']:
                                        if args.useatypiaclasscombiners in ["True", "true", "TRUE"]:
                                            tileDictionary['artifact'].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                        else:
                                            tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                    elif annotation.attrib['PartOfGroup'] in tileDictionary:
                                        tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                    else:
                                        raise Warning('Annotation ' + annotation.attrib['PartOfGroup'] + ' is not part of a pre-defined group')

                                else: # p53
                                #aberrant_positive_columnar+equivocal_columnar, artifact+nonspecific_background+oral_bacteria
                                    if annotation.attrib['PartOfGroup'] in ['equivocal_columnar']:
                                        if args.usep53classcombiners in ["True", "true", "TRUE"]:
                                            continue
                                            #tileDictionary['aberrant_positive_columnar'].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                        else:
                                            tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                    elif annotation.attrib['PartOfGroup'] in ['nonspecific_background', 'oral_bacteria']:
                                        if args.usep53classcombiners in ["True", "true", "TRUE"]:
                                            tileDictionary['artifact'].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                        else:
                                            tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                    elif annotation.attrib['PartOfGroup'] in ['respiratory_mucosa']:
                                        if args.usep53classcombiners in ["True", "true", "TRUE"]:
                                            tileDictionary['wild_type_columnar'].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                        else:
                                            tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                    elif annotation.attrib['PartOfGroup'] in tileDictionary:
                                        tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                    else:
                                        raise Warning('Annotation ' + annotation.attrib['PartOfGroup'] + ' is not part of a pre-defined group')





                                #if args.combineatypia in ["True", "true", "TRUE"]:
                                #    if (annotation.attrib['PartOfGroup'] not in ['atypia_of_uncertain_significance', 'dysplasia']) and (annotation.attrib['PartOfGroup'] in tileDictionary):
                                #        tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                #    elif annotation.attrib['PartOfGroup'] in ['atypia_of_uncertain_significance', 'dysplasia']:
                                #        tileDictionary['atypia'].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                #    else:
                                #        raise Warning('Annotation ' + annotation.attrib['PartOfGroup'] + ' is not part of a pre-defined group')
                                # Do not combine atypia_of_uncertain_significance and dysplasia tiles into one class; keep them separate
                                #else:
                                #    if annotation.attrib['PartOfGroup'] in tileDictionary:
                                #        tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                                #    else:
                                #        raise Warning('Annotation ' + annotation.attrib['PartOfGroup'] + ' is not part of a pre-defined group')

                # The annotation is only large enough to extract one tile from it
                else:
                    minX = geometry.Point(poly.centroid).coords.xy[0][0]-tileSize/2
                    minY = geometry.Point(poly.centroid).coords.xy[1][0]-tileSize/2
                    maxX = geometry.Point(poly.centroid).coords.xy[0][0]+tileSize/2
                    maxY = geometry.Point(poly.centroid).coords.xy[1][0]+tileSize/2
                    tileBox = geometry.box(minX,minY,maxX,maxY)
                    intersectingArea = poly.intersection(tileBox).area/tileSize**2

                    if whichStain == 'he':
                        # combine atypia_of_uncertain_significance and dysplasia tiles into one class: atypia
                        if annotation.attrib['PartOfGroup'] in ['atypia_of_uncertain_significance', 'dysplasia']:
                            if args.useatypiaclasscombiners in ["True", "true", "TRUE"] or args.combineatypia in ["True", "true", "TRUE"]:
                                tileDictionary['atypia'].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                            else:
                                tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                        elif annotation.attrib['PartOfGroup'] in ['respiratory_mucosa_cilia']:
                            if args.useatypiaclasscombiners in ["True", "true", "TRUE"] or args.combinerespiratory in ["True", "true", "TRUE"]:
                                tileDictionary['respiratory_mucosa'].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                            else:
                                tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                        elif annotation.attrib['PartOfGroup'] in ['tickled_up_columnar', 'IM_cardia']:
                            if args.useatypiaclasscombiners in ["True", "true", "TRUE"] or args.combinegastriccardia in ["True", "true", "TRUE"]:
                                tileDictionary['gastric_cardia'].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                            else:
                                tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                        elif annotation.attrib['PartOfGroup'] in ['other']:
                            if args.useatypiaclasscombiners in ["True", "true", "TRUE"]:
                                tileDictionary['artifact'].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                            else:
                                tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                        elif annotation.attrib['PartOfGroup'] in tileDictionary:
                            tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                        else:
                            raise Warning('Annotation ' + annotation.attrib['PartOfGroup'] + ' is not part of a pre-defined group')

                    else: # p53
                    #aberrant_positive_columnar+equivocal_columnar, artifact+nonspecific_background+oral_bacteria
                        if annotation.attrib['PartOfGroup'] in ['equivocal_columnar']:
                            if args.usep53classcombiners in ["True", "true", "TRUE"]:
                                continue
                                #tileDictionary['aberrant_positive_columnar'].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                            else:
                                tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                        elif annotation.attrib['PartOfGroup'] in ['nonspecific_background', 'oral_bacteria']:
                            if args.usep53classcombiners in ["True", "true", "TRUE"]:
                                tileDictionary['artifact'].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                            else:
                                tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                        elif annotation.attrib['PartOfGroup'] in ['respiratory_mucosa']:
                            if args.usep53classcombiners in ["True", "true", "TRUE"]:
                                tileDictionary['wild_type_columnar'].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                            else:
                                tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                        elif annotation.attrib['PartOfGroup'] in tileDictionary:
                            tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                        else:
                            raise Warning('Annotation ' + annotation.attrib['PartOfGroup'] + ' is not part of a pre-defined group')



                    #if args.combineatypia in ["True", "true", "TRUE"]:
                    #    if (annotation.attrib['PartOfGroup'] not in ['atypia_of_uncertain_significance', 'dysplasia']) and (annotation.attrib['PartOfGroup'] in tileDictionary):
                    #        tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                    #    elif annotation.attrib['PartOfGroup'] in ['atypia_of_uncertain_significance', 'dysplasia']:
                    #        tileDictionary['atypia'].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                    #    else:
                    #        raise Warning('Annotation is not part of a pre-defined group: '+annotation.attrib['PartOfGroup'])
                    # Do not combine atypia_of_uncertain_significance and dysplasia tiles into one class; keep them separate
                    #else:
                    #    if annotation.attrib['PartOfGroup'] in tileDictionary:
                    #        tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
                    #    else:
                    #        raise Warning('Annotation is not part of a pre-defined group')

        pickle.dump(tileDictionary, open(os.path.join(outputFolderPath, slideFileName.replace('.'+wsi_extension, '')+'.p'), 'wb'))

    else:
        raise Warning(best2Id+', for which an annotation file exists, is not among list of pathologist verified '+whichStain.upper()+' annotations')
