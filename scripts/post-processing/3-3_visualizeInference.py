# 3-3_visualizeInference.py
#
# Adam Berman and Marcel Gehrung
#
# This file visualizes the inference of a model onto a slide or slides
#


import os
import argparse
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import imageio, sys
import pyvips as pv
from skimage.filters import gaussian
from skimage.transform import resize
sys.path.append('/home/cri.camres.org/berman01/pathml') # install from here: https://github.com/9xg/pathml
from pathml import slide
import argparse


parser = argparse.ArgumentParser(description='Plot inference tiles')
parser.add_argument("-stain", required=True, help="he or p53")
#parser.add_argument("-combineatypia", required=True, help="True or False as to whether to combine the atypia of uncertain significance and dysplasia classes")
parser.add_argument("-slide", default='all_inference_folder', help="path to WSI if making map for single slide; set to 'all_inference_folder' to infer on the whole inference folder or 'all_test_set' to infer on the whole test set")
parser.add_argument("-inferencefolder", required=True, help="path to directory containing inference file(s)")
parser.add_argument("-slidefolder", required=True, help="path to directory containing WSIs")
parser.add_argument("-output", required=True, help="path to directory to save inference map(s) to")
parser.add_argument("-whichclass", default=False, help="which class to make inference maps for; if not defined, all class's inference maps will be created")
parser.add_argument("-combineatypia", default="True", help="True or False as to whether to combine the atypia of uncertain significance and dysplasia classes (default is True)")
parser.add_argument("-combinerespiratory", default="True", help="True or False as to whether to combine the respiratory mucosa cilia and respiratory mucosa classes (default is True)")
parser.add_argument("-combinegastriccardia", default="True", help="True or False as to whether to combine the tickled up columnar and gastric cardia classes (default is True)")
parser.add_argument("-useatypiaclasscombiners", default="True", help="True or False as to whether to perform the following class mergers: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other (default is True)")
parser.add_argument("-usep53classcombiners", default="True", help="True or False as to whether to perform the following class mergers: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar (default is True)")
parser.add_argument("-extractdiseasetiles", default=None, help="A threshold above or equal to disease tiles (atypia tiles for H&E, aberrant P53 columnar for P53) are extracted to the output folder. Default is not to extract these tiles.")
args = parser.parse_args()

if args.stain not in ['he', 'p53']:
    raise Warning("-stain argument must be either 'he' or 'p53'")
if args.stain == 'he':
    disease_class = 'atypia'
else:
    disease_class = 'aberrant_positive_columnar'

if args.slide == "all_test_set": # make map for all test set slides
    slideFile = []
    #inferenceFile = []

    best2GroundTruth = pd.read_excel('data/atypia-and-p53-ground-truth-labels/best2_test_set_consensus_atypia_and_p53_labels.xlsx')
    best2GroundTruth['SampleID_2'] = best2GroundTruth['SampleID_2'].astype(str).str.replace('/', ' ')
    best3GroundTruth = pd.read_excel('data/atypia-and-p53-ground-truth-labels/best3_test_set_consensus_atypia_and_p53_labels.xlsx')
    best3GroundTruth['BEST3_ID'] = best3GroundTruth['BEST3_ID'].astype(str).str.replace('.', ' ')

    # Identify cases current test dataset lacks slides for
    test_cases_with_gt = [x for x in best2GroundTruth['SampleID_2'].tolist() + best3GroundTruth['BEST3_ID'].tolist() if x != 'nan']
    test_cases_with_gt.sort()
    test_cases_with_he_slide_paths = glob.glob(os.path.join("/media/berman01/Seagate Expansion Drive/test-set-slides/he-slides", "*.ndpi"))
    test_cases_with_he_slide_paths.sort()
    test_cases_with_p53_slide_paths = glob.glob(os.path.join("/media/berman01/Seagate Expansion Drive/test-set-slides/p53-slides", "*.ndpi"))
    test_cases_with_p53_slide_paths.sort()
    trainval_cases_with_he_slide_paths = glob.glob(os.path.join("/media/berman01/Backup Plus/trainval-set-slides/he-slides", "*.xml"))
    trainval_cases_with_he_slide_paths.sort()
    trainval_cases_with_p53_slide_paths = glob.glob(os.path.join("/media/berman01/Backup Plus/trainval-set-slides/p53-slides", "*.xml"))
    trainval_cases_with_p53_slide_paths.sort()
    test_cases_with_he_slide = [os.path.split(tc_path)[-1].split(" HE")[0] for tc_path in test_cases_with_he_slide_paths]
    test_cases_with_p53_slide = [os.path.split(tc_path)[-1].split(" P53")[0] for tc_path in test_cases_with_p53_slide_paths]
    trainval_cases_with_he_slide = [os.path.split(tc_path)[-1].split(" HE")[0] for tc_path in trainval_cases_with_he_slide_paths]
    trainval_cases_with_p53_slide = [os.path.split(tc_path)[-1].split(" P53")[0] for tc_path in trainval_cases_with_p53_slide_paths]
    test_cases_with_he_slide.sort()
    test_cases_with_p53_slide.sort()
    #print("Test cases with ground truth (n="+str(len(test_cases_with_gt))+"):")
    #print(test_cases_with_gt, "\n")
    #print("Test cases with HE slide (n="+str(len(test_cases_with_he_slide))+"):")
    #print(test_cases_with_he_slide, "\n")
    #print("Test cases with P53 slide (n="+str(len(test_cases_with_p53_slide))+"):")
    #print(test_cases_with_p53_slide)

    for case in test_cases_with_he_slide_paths:
        #print(case)
        caseId = os.path.split(case)[-1].split(" "+args.stain.upper())[0]
        #print(caseId)
        #quit()
        #caseIdSlash = '/'.join(os.path.split(case)[-1].replace('.p', '').split('_')[0:3])

        # only add prediction if we have a ground truth label for the patient and that patient does not appear in the trainval set
        if ((caseId in best2GroundTruth['SampleID_2'].values) or (caseId in best3GroundTruth['BEST3_ID'].values)) and (caseId not in trainval_cases_with_he_slide) and (caseId not in trainval_cases_with_p53_slide):
            #continue
            #cases.append(case.replace('.'+args.wsiextension, ''))
            slideFile.append(case)
            #inferenceFile.append(os.path.join(args.inferencepath, os.path.split(case)[-1].replace(".ndpi", "_inference.p")))

elif args.slide == 'all_inference_folder':
    inferences = glob.glob(os.path.join(args.inferencefolder, "*_inference.p"))
    inference_names = [os.path.split(inference)[-1].replace("_inference.p", ".ndpi") for inference in inferences]
    slideFile = []
    for inference_name in inference_names:
        if os.path.exists(os.path.join(args.slidefolder, inference_name)):
            slideFile.append(os.path.join(args.slidefolder, inference_name))
        else:
            raise Warning(inference_name+' not present in '+args.slidefolder)

        '''
        if os.path.exists(os.path.join('/media/berman01/Backup Plus/trainval-set-slides/he-slides', inference_name)):
            slideFile.append(os.path.join('/media/berman01/Backup Plus/trainval-set-slides/he-slides', inference_name))
        #elif os.path.exists(os.path.join('/media/berman01/Seagate Expansion Drive/test-set-slides/he-slides', inference_name)):
        elif os.path.exists(os.path.join('/media/berman01/Seagate Expansion Drive/ATYPIA-DETECTION-SLIDES/testing-slides', inference_name)):
            #slideFile.append(os.path.join('/media/berman01/Seagate Expansion Drive/test-set-slides/he-slides', inference_name))
            slideFile.append(os.path.join('/media/berman01/Seagate Expansion Drive/ATYPIA-DETECTION-SLIDES/testing-slides', inference_name))
    #slideFile = [glob.glob(os.path.join(args.inferencefolder, ))]
        '''

else: # mape map for individual slide
    slideFile = [args.slide]
    #inferenceFile = [args.inferencepath]

#print(slideFile)
#quit()
slideFile.sort()

inferenceFile = [os.path.join(args.inferencefolder, os.path.split(sf)[-1].replace(".ndpi", "_inference.p")) for sf in slideFile]
#print(inferenceFile)
#quit()

#tileDictionary = pickle.load(open('BEST2_UCL_0101_HE_1.p', "rb"))
#tileDictionary = pickle.load(open('BEST2_CAM_0008_HE_1.p', "rb"))
#caseToShow = os.path.split(args.slide)[-1].replace("_inference.p", "")
caseToShow = [os.path.split(sf)[-1].replace(".ndpi", "") for sf in slideFile]


#pathSlide = [slide.Slide(sf) for sf in slideFile]
#ourNewImg = [ps.thumbnail(level=4) for ps in pathSlide]
#ourNewImg=[]

#tileDictionary = [pickle.load(open(infi, "rb")) for infi in inferenceFile]

if args.stain == "he":
    if args.useatypiaclasscombiners in ["True", "true", "TRUE"]:
        class_names = ['artifact', 'atypia', 'background', 'gastric_cardia', 'immune_cells',
                        'intestinal_metaplasia', 'respiratory_mucosa', 'squamous_mucosa']

    else:

        if args.combineatypia in ["True", "true", "TRUE"]:
            if args.combinerespiratory in ["True", "true", "TRUE"]:
                if args.combinegastriccardia in ["True", "true", "TRUE"]:
                    class_names = ['artifact', 'atypia', 'background', 'gastric_cardia', 'immune_cells',
                                    'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'squamous_mucosa']
                else:
                    class_names = ['artifact', 'atypia', 'background', 'gastric_cardia', 'immune_cells',
                                    'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'squamous_mucosa', 'tickled_up_columnar']
            else:
                if args.combinegastriccardia in ["True", "true", "TRUE"]:
                    class_names = ['artifact', 'atypia', 'background', 'gastric_cardia', 'immune_cells',
                                    'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'respiratory_mucosa_cilia', 'squamous_mucosa']
                else:
                    class_names = ['artifact', 'atypia', 'background', 'gastric_cardia', 'immune_cells',
                                    'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'respiratory_mucosa_cilia', 'squamous_mucosa', 'tickled_up_columnar']

        else:
            if args.combinerespiratory in ["True", "true", "TRUE"]:
                if args.combinegastriccardia in ["True", "true", "TRUE"]:
                    class_names = ['artifact', 'atypia_of_uncertain_significance', 'background', 'dysplasia', 'gastric_cardia',
                                    'immune_cells', 'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'squamous_mucosa']
                else:
                    class_names = ['artifact', 'atypia_of_uncertain_significance', 'background', 'dysplasia', 'gastric_cardia',
                                    'immune_cells', 'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'squamous_mucosa', 'tickled_up_columnar']
            else:
                if args.combinegastriccardia in ["True", "true", "TRUE"]:
                    class_names = ['artifact', 'atypia_of_uncertain_significance', 'background', 'dysplasia', 'gastric_cardia',
                                    'immune_cells', 'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'respiratory_mucosa_cilia', 'squamous_mucosa']
                else:
                    class_names = ['artifact', 'atypia_of_uncertain_significance', 'background', 'dysplasia', 'gastric_cardia',
                                    'immune_cells', 'intestinal_metaplasia', 'other', 'respiratory_mucosa', 'respiratory_mucosa_cilia', 'squamous_mucosa', 'tickled_up_columnar']
elif args.stain == "p53":
    if args.usep53classcombiners:
        class_names = ['aberrant_positive_columnar', 'artifact', 'background', 'immune_cells', 'squamous_mucosa', 'wild_type_columnar']
    else:
        class_names = ['aberrant_positive_columnar', 'artifact', 'background', 'equivocal_columnar',
                        'immune_cells', 'nonspecific_background', 'oral_bacteria', 'respiratory_mucosa', 'squamous_mucosa', 'wild_type_columnar']

if args.whichclass:
    classes_to_make_mask_for = [args.whichclass]
else:
    classes_to_make_mask_for = class_names

#class_masks = {}
#class_masks = []
#for td in tileDictionary:
#    cm = {}
#    for class_to_make_mask_for in classes_to_make_mask_for:
#        cm[class_to_make_mask_for] = np.zeros(td['maskSize'][::-1])
#    class_masks.append(cm)

#equivPredictionImage = np.zeros(tileDictionary['maskSize'][::-1])
#positivePredictionImage = np.zeros(tileDictionary['maskSize'][::-1])
#negativePredictionImage = np.zeros(tileDictionary['maskSize'][::-1])
#foregroundImage = np.zeros(tileDictionary['maskSize'][::-1])

#print(predictionImage.shape)
#for i, td in enumerate(tileDictionary):
for i, infi in enumerate(inferenceFile):

    pathSlide = slide.Slide(slideFile[i])
    ourNewImg = pathSlide.thumbnail(level=4)

    td = pickle.load(open(infi, "rb"))

    #class_masks = []
    #for td in tileDictionary:
    class_masks = {}
    for class_to_make_mask_for in classes_to_make_mask_for:
        class_masks[class_to_make_mask_for] = np.zeros(td['maskSize'][::-1])
    #class_masks.append(cm)

    sliname = os.path.split(slideFile[i])[-1].replace(".ndpi", "")

    if args.extractdiseasetiles:
        heSlide = pv.Image.new_from_file(
            slideFile[i], level=0)
        #print('Making disease tiles (if any) for '+sliname)

    diseased_tiles_dict = {}

    for key,val in td['tileDictionary'].items():

        for idx, class_name in enumerate(class_names):

            # extract disease tiles
            if args.extractdiseasetiles:
                if class_name == disease_class:
                    disease_prob = val['prediction'][idx] if 'prediction' in val else 0
                    if disease_prob >= float(args.extractdiseasetiles): # tile is considered diseased

                        # SAVE DISEASE TILE ASAP ANNOTATION FILE
                        diseased_tiles_dict[str(disease_prob)+'_'+str(val['x'])+'_'+str(val['y'])] = {disease_class+'_probability': disease_prob, 'x': val['x'], 'y': val['y'], 'width': val['width']}

                        # SAVE DISEASE TILES
                        #print('val', val)
                        if not os.path.exists(os.path.join(args.output, disease_class+'_tiles', sliname, disease_class + '_' + str(disease_prob) + '_' + str(
                            val['x']) + '_' + str(val['y']) + '_' + str(val['width']) + '.jpg')):
                            try:
                                areaHe = heSlide.extract_area(
                                    val['x'], val['y'], val['width'], val['height'])
                                os.makedirs(os.path.join(args.output, disease_class+'_tiles', sliname), exist_ok=True)
                                areaHe.write_to_file(os.path.join(args.output, disease_class+'_tiles', sliname, disease_class + '_' + str(disease_prob) + '_' + str(
                                    val['x']) + '_' + str(val['y']) + '_' + str(val['width']) + '.jpg'), Q=100)
                                print('Made disease tile')
                            except:
                                print('Skipping tile that goes beyond the edge of the WSI...')
                        else:
                            print('Disease tile already exists, skipping...')
                #continue

            if (args.whichclass):
                if (class_name == args.whichclass):
                    class_masks[class_name][key[1],key[0]] = val['prediction'][idx] if 'prediction' in val else 0
                    #class_masks[i][class_name][key[1],key[0]] = val['prediction'][idx] if 'prediction' in val else 0
            else:
                class_masks[class_name][key[1],key[0]] = val['prediction'][idx] if 'prediction' in val else 0
                #class_masks[i][class_name][key[1],key[0]] = val['prediction'][idx] if 'prediction' in val else 0


        #equivPredictionImage[key[1],key[0]] = val['prediction'][0] if 'prediction' in val else 0
        #negativePredictionImage[key[1],key[0]] = val['prediction'][1] if 'prediction' in val else 0
        #positivePredictionImage[key[1],key[0]] = val['prediction'][2] if 'prediction' in val else 0
        #foregroundImage[key[1],key[0]] = int(val['foreground'] if 'foreground' in val else 0)

#if not os.path.exists(args.output):
#    os.makedirs(args.output)

#for i, cts in enumerate(caseToShow):
#    imageio.imwrite(os.path.join(args.output, cts+'_'+args.stain+'.png'), resize(ourNewImg[i], np.multiply(class_masks[i][class_names[0]].shape,3)))
#quit()

    #for i, cm in enumerate(class_masks):
    #sliname = os.path.split(slideFile[i])[-1].replace(".ndpi", "")
    #if os.path.exists(os.path.join(args.output, sliname)):
    #    print("Visual inference for "+sliname+" already exists. Skipping.")
    #    continue
    #else:
    #    os.makedirs(os.path.join(args.output, sliname))


    # Make ASAP file
    xml_header = """<?xml version="1.0"?>
<ASAP_Annotations>
\t<Annotations>\n"""

    xml_tail = 	"""\t</Annotations>
\t<AnnotationGroups>
\t\t<Group Name="atypia" PartOfGroup="None" Color="#64FE2E">
\t\t\t<Attributes />
\t\t</Group>
\t</AnnotationGroups>
</ASAP_Annotations>\n"""
    xml_tail = xml_tail.replace('atypia', disease_class)

    xml_annotations = ""

    if diseased_tiles_dict:
        if not os.path.exists(os.path.join(args.output, disease_class+'_tile_annotations', sliname+'_inference_'+disease_class+'.xml')):
            os.makedirs(os.path.join(args.output, disease_class+'_tile_annotations'), exist_ok=True)
            for key, tile_info in sorted(diseased_tiles_dict.items(), reverse=True):
        #            xml_annotations = xml_annotations + """\t\t<Annotation Name="Annotation 0" Type="Rectangle" PartOfGroup="atypia" Color="#F4FA58">
        #\t\t\t<Coordinates>
        #\t\t\t\t<Coordinate Order="0" X="25722.3223" Y="33845.7812" />
        #\t\t\t\t<Coordinate Order="1" X="26401.2266" Y="33845.7812" />
        #\t\t\t\t<Coordinate Order="2" X="26401.2266" Y="34384.2188" />
        #\t\t\t\t<Coordinate Order="3" X="25722.3223" Y="34384.2188" />
        #\t\t\t</Coordinates>
        #\t\t</Annotation>\n"""
                xml_annotations = (xml_annotations +
                                    "\t\t<Annotation Name=\""+str(tile_info[disease_class+'_probability'])+"\" Type=\"Polygon\" PartOfGroup=\""+disease_class+"\" Color=\"#F4FA58\">\n" +
                                    "\t\t\t<Coordinates>\n" +
                                    "\t\t\t\t<Coordinate Order=\"0\" X=\""+str(tile_info['x'])+"\" Y=\""+str(tile_info['y'])+"\" />\n" +
                                    "\t\t\t\t<Coordinate Order=\"1\" X=\""+str(tile_info['x']+tile_info['width'])+"\" Y=\""+str(tile_info['y'])+"\" />\n" +
                                    "\t\t\t\t<Coordinate Order=\"2\" X=\""+str(tile_info['x']+tile_info['width'])+"\" Y=\""+str(tile_info['y']+tile_info['width'])+"\" />\n" +
                                    "\t\t\t\t<Coordinate Order=\"3\" X=\""+str(tile_info['x'])+"\" Y=\""+str(tile_info['y']+tile_info['width'])+"\" />\n" +
                                    "\t\t\t</Coordinates>\n" +
                                    "\t\t</Annotation>\n")
            print('Creating automated annotation file for '+sliname)
            with open(os.path.join(args.output, disease_class+'_tile_annotations', sliname+'_inference_'+disease_class+'.xml'), "w") as text_file:
                text_file.write(xml_header + xml_annotations + xml_tail)
            #quit()
        else:
            print('Automated annotation file already exists...')


        #print(xml_annotations)








    # SAVE INFERENCE MAP

    os.makedirs(os.path.join(args.output, sliname), exist_ok=True)
    print("Visualizing inference for "+sliname+"...")
    for idx, class_name in enumerate(classes_to_make_mask_for):
        if os.path.isfile(os.path.join(args.output, sliname, sliname+"_"+class_name+".png")):
            print("Visual inference of "+class_name+" class for "+sliname+" already exists. Skipping.")
            continue
        plt.figure()
        #plt.imshow(foregroundImage,cmap='Greys')
        plt.imshow(resize(ourNewImg, class_masks[class_name].shape))
        plt.imshow(class_masks[class_name],cmap='plasma',alpha=0.3, vmin=0, vmax=1.0)
        plt.colorbar()
        plt.title(sliname+"\n"+class_name)
        plt.savefig(os.path.join(args.output, sliname, sliname+"_"+class_name+".png"))
        plt.clf()
    #print("Inference visualization complete for "+sliname)
        #if idx != len(class_names)-1:
        #    plt.show(block=False)
        #else:
        #    plt.show()

'''
plt.figure()
#plt.imshow(foregroundImage,cmap='Greys')
plt.imshow(resize(ourNewImg, equivPredictionImage.shape))
plt.imshow(equivPredictionImage,cmap='plasma',alpha=0.3, vmin=0, vmax=0.4)
plt.colorbar()
plt.title('Equivocal likelihood')
plt.show(block=False)

plt.figure()
#plt.imshow(foregroundImage,cmap='Greys')
plt.imshow(resize(ourNewImg, equivPredictionImage.shape))
plt.imshow(positivePredictionImage,cmap='plasma',alpha=0.3, vmin=0, vmax=0.5)
plt.colorbar()
plt.title('Positive likelihood')
plt.show(block=False)

plt.figure()
#plt.imshow(foregroundImage,cmap='Greys')
plt.imshow(resize(ourNewImg, equivPredictionImage.shape))
plt.imshow(negativePredictionImage,cmap='plasma',alpha=0.3, vmin=0, vmax=1)
plt.colorbar()
plt.title('Negative likelihood')
plt.show()
'''
