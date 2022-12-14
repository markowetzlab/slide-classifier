# 3-1_hePredictionsToDataframe.py
#
# Adam Berman and Marcel Gehrung
#
# This script converts tile predictions of individual whole-slide images to aggregation dataframes
#

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import imageio
from skimage.filters import gaussian
from tqdm import tqdm
import argparse
import glob
import os

parser = argparse.ArgumentParser(description='Convert predictions into differential counts.')
parser.add_argument("-stain", required=True, help="he or p53")
parser.add_argument("-inferencemapsrootfolder", required=True, help="root folder of inference maps")
parser.add_argument("-combineatypia", default="True", help="True or False as to whether to combine the atypia of uncertain significance and dysplasia classes (default is True)")
parser.add_argument("-combinerespiratory", default="True", help="True or False as to whether to combine the respiratory mucosa cilia and respiratory mucosa classes (default is True)")
parser.add_argument("-combinegastriccardia", default="True", help="True or False as to whether to combine the tickled up columnar and gastric cardia classes (default is True)")
parser.add_argument("-useatypiaclasscombiners", default="True", help="True or False as to whether to perform the following class mergers: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other (default is True)")
parser.add_argument("-usep53classcombiners", default="True", help="True or False as to whether to perform the following class mergers: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar (default is True)")
parser.add_argument("-output", help="folder to store completed data frame to; default is -inferencemapsrootfolder")
parser.add_argument("-rankedtileclass", default="DEFAULT", help="class to make ranked tile CSV for (default is 'atypia' if stain is 'he' or 'aberrant_positive_columnar' if stain is 'p53')")
parser.add_argument("-groundtruth", default='data/just_labels.csv', help="file containing slide-level ground truth to use. default is 'data/just_labels.csv'")
#parser.add_argument("-slidestouse", required=True, help="'test' for test set; 'val' for val set")
args = parser.parse_args()

inferenceFolder = args.inferencemapsrootfolder

if args.output:
    outputPath = args.output
else:
    outputPath = args.inferencemapsrootfolder
#whichArchitecture = args.architecture

if args.stain not in ['he', 'p53']:
    raise Warning("-stain argument must be either 'he' or 'p53'")

if args.rankedtileclass == 'DEFAULT':
    if args.stain == 'he':
        rankedTileClass = 'atypia'
    else:
        rankedTileClass = 'aberrant_positive_columnar'
else:
    rankedTileClass = args.rankedtileclass

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

if args.output:
    if not os.path.exists(args.output):
        os.makedirs(args.output)

#probabilitiesToThreshold = np.append(np.arange(0, 1, 0.005),[0.999,0.9999,0.99999])
probabilitiesToThreshold = np.append(np.arange(0, 0.99, 0.005), np.arange(0.99, 0.9989, 0.001))#0.0005)) #np.arange(0.99, 0.9989, 0.001))
probabilitiesToThreshold = np.append(probabilitiesToThreshold, np.arange(0.999, 0.99989, 0.0001))
probabilitiesToThreshold = np.append(probabilitiesToThreshold, np.arange(0.9999, 0.999989, 0.00001))
probabilitiesToThreshold = np.append(probabilitiesToThreshold, np.arange(0.99999, 0.9999989, 0.000001))
probabilitiesToThreshold = np.append(probabilitiesToThreshold, np.arange(0.999999, 1, 0.0000001))
#[0.9925, 0.995, 0.9975, 0.999, 0.99925, 0.9995, 0.99975, 0.9999, 0.999925, 0.99995, 0.999975, 0.99999, 0.9999925, 0.999995, 0.9999975, 0.999999]
#print(probabilitiesToThreshold)
#print(len(probabilitiesToThreshold))
#quit()

ground_truth = pd.read_csv(args.groundtruth)

#if args.slidestouse == "test":
#    best2GroundTruth = pd.read_excel('data/atypia-and-p53-ground-truth-labels/best2_test_set_consensus_atypia_and_p53_labels.xlsx')
#    best2GroundTruth['SampleID_2'] = best2GroundTruth['SampleID_2'].astype(str).str.replace('/', ' ')
#    print(best2GroundTruth['SampleID_2'].tolist())
#    best3GroundTruth = pd.read_excel('data/atypia-and-p53-ground-truth-labels/best3_test_set_consensus_atypia_and_p53_labels.xlsx')
#    #best3GroundTruth['BEST3_ID'] = best3GroundTruth['BEST3_ID'].astype(str)
#    best3GroundTruth['BEST3_ID'] = best3GroundTruth['BEST3_ID'].astype(str).str.replace('.', ' ')
#best3GroundTruth['BEST3_ID'] = 'TB' + best3GroundTruth['BEST3_ID']

    # Identify cases current test dataset lacks slides for
#    test_cases_with_gt = [x for x in best2GroundTruth['SampleID_2'].tolist() + best3GroundTruth['BEST3_ID'].tolist() if x != 'nan']
#    test_cases_with_gt.sort()
#    test_cases_with_he_slide_paths = glob.glob(os.path.join("/media/berman01/Seagate Expansion Drive/test-set-slides/he-slides", "*.ndpi"))
#    test_cases_with_p53_slide_paths = glob.glob(os.path.join("/media/berman01/Seagate Expansion Drive/test-set-slides/p53-slides", "*.ndpi"))
#    trainval_cases_with_he_slide_paths = glob.glob(os.path.join("/media/berman01/Backup Plus/trainval-set-slides/he-slides", "*.xml"))
#    trainval_cases_with_p53_slide_paths = glob.glob(os.path.join("/media/berman01/Backup Plus/trainval-set-slides/p53-slides", "*.xml"))
#    test_cases_with_he_slide = [os.path.split(tc_path)[-1].split(" HE")[0] for tc_path in test_cases_with_he_slide_paths]
#    test_cases_with_p53_slide = [os.path.split(tc_path)[-1].split(" P53")[0] for tc_path in test_cases_with_p53_slide_paths]
#    trainval_cases_with_he_slide = [os.path.split(tc_path)[-1].split(" HE")[0] for tc_path in trainval_cases_with_he_slide_paths]
#    trainval_cases_with_p53_slide = [os.path.split(tc_path)[-1].split(" P53")[0] for tc_path in trainval_cases_with_p53_slide_paths]
#    test_cases_with_he_slide.sort()
#    test_cases_with_p53_slide.sort()
#    print("Test cases with ground truth (n="+str(len(test_cases_with_gt))+"):")
#    #print(test_cases_with_gt, "\n")
#    print("Test cases with HE slide (n="+str(len(test_cases_with_he_slide))+"):")
#    #print(test_cases_with_he_slide, "\n")
#    print("Test cases with P53 slide (n="+str(len(test_cases_with_p53_slide))+"):")
#    #print(test_cases_with_p53_slide)

#    for tcwgt in test_cases_with_gt:
#        if (tcwgt not in trainval_cases_with_he_slide) and (tcwgt not in trainval_cases_with_p53_slide):
#            if tcwgt not in test_cases_with_he_slide:
#                print(tcwgt + " has ground truth but no H&E slide")
#            if tcwgt not in test_cases_with_p53_slide:
#                print(tcwgt + " has ground truth but no P53 slide")
#        else:
#            print(tcwgt + " APPEARS IN HE TRAINVAL SET")

#quit()


#print(best2GroundTruth)
#print(best3GroundTruth)
#quit()
#cytospongeData.dropna(subset=['Gland groups on HE','TFF3+'],inplace=True)
#cytospongeData.replace({'Gland groups on HE': {'>5': 6}}, inplace=True)
#endoscopyData.replace({'PRAGUE_M': {'NR': 0, '<1': 0.5}, 'PRAGUE_C': {'NR': 0, '<1': 0.5}}, inplace=True)
print('Run '+args.stain.upper()+' predictions to dataframe conversion')



rows_list = []
rows_list_rankedtiles = []
atypia_positive_ground_truth_counter = 0
p53_positive_ground_truth_counter = 0
inferenceMapPaths = glob.glob(os.path.join(inferenceFolder, '*_inference.p'))
inferenceMapPaths.sort()
for inferenceMapPath in inferenceMapPaths:#tqdm(inferenceMapPaths):
    #print(case)
    ####caseId = os.path.split(case)[-1].replace('_inference.p', '').split(' '+args.stain.upper())[0]

    # go from inference map path to this format: "BEST2/CAM/0001" or "BEST2/CAM/0001_R1"
    caseID = os.path.split(inferenceMapPath)[-1].split(' '+args.stain.upper()+' ')[0]
    caseIDSplit = caseID.split(' ')
    caseID = ''
    numPieces = len(caseIDSplit)
    for i, casePiece in enumerate(caseIDSplit):
        if 'P53' in casePiece or 'HE' in casePiece:
            break
        if i == 0:
            caseID = casePiece
        elif 'R' in casePiece and i == (numPieces-1):
            caseID = caseID + '_' + casePiece
        else:
            caseID = caseID + '/' + casePiece

    #print(caseID)
    #quit()

    #print(caseId)
    #quit()
    #caseIdSlash = '/'.join(os.path.split(case)[-1].replace('.p', '').split('_')[0:3])

#    if args.slidestouse == 'test':
#
#        # only add prediction if we have a ground truth label for the patient and that patient does not appear in the trainval set
#        if ((caseId in best2GroundTruth['SampleID_2'].values) or (caseId in best3GroundTruth['BEST3_ID'].values)) and (caseId not in trainval_cases_with_he_slide) and (caseId not in trainval_cases_with_p53_slide):
#            #continue
#            print("Case ID: "+caseId)
#            #quit()
#            tileDictionary = pickle.load(open(case, "rb"))
#            predictionImages = {class_name: np.zeros(tileDictionary['maskSize'][::-1]) for class_name in class_names}
#
#            for key, val in tileDictionary['tileDictionary'].items():
#
#                for class_idx, class_name in enumerate(class_names):
#                    predictionImages[class_name][key[1], key[0]] = val['prediction'][class_idx] if 'prediction' in val else 0

                #respPredictionImage[key[1], key[0]] = val['prediction'][3] if 'prediction' in val else 0
                #gastPredictionImage[key[1], key[0]] = val['prediction'][1] if 'prediction' in val else 0
                #imPredictionImage[key[1], key[0]] = val['prediction'][2] if 'prediction' in val else 0

#            if caseId in best2GroundTruth['SampleID_2'].values:
#                if caseId in ["BEST2 CAM 0045", "BEST2 CAM 0085", "BEST2 CAM 0670", "BEST2 UCL 0048"]:
#                    atypiaGroundTruth = 0
#                    p53GroundTruth = 0
#                elif caseId in ["BEST2 UCL 0034", "BEST2 UCL 0038"]:
#                    atypiaGroundTruth = 1
#                    p53GroundTruth = 0
#                elif caseId in ["BEST2 UCL 0032", "BEST2 UCL 0042", "BEST2 UCL 0057"]:
#                    atypiaGroundTruth = 1
#                    p53GroundTruth = 1
#                else:
#                    atypiaGroundTruth = int(best2GroundTruth.loc[(best2GroundTruth['SampleID_2'] == caseId)]['Atypiafinal']) #Atypiafinal
#                    p53GroundTruth = int(best2GroundTruth.loc[(best2GroundTruth['SampleID_2'] == caseId)]['p53final']) #p53final
#            else:
#                atypiaGroundTruth = int(best3GroundTruth.loc[(best3GroundTruth['BEST3_ID'] == caseId)]['Cytosponge_atypia']) #Cytosponge_atypia
#                p53GroundTruth = int(best3GroundTruth.loc[(best3GroundTruth['BEST3_ID'] == caseId)]['Cytosponge_p53']) #Cytosponge_p53

#            print("Atypia ground truth: "+str(atypiaGroundTruth))
#            print("P53 ground truth: "+str(p53GroundTruth))

#            countsAboveProbability = []
#            countsAboveProbabilityColumnNames = []
#            for class_name in class_names:
#                countsAboveProbability = countsAboveProbability + [np.count_nonzero(predictionImages[class_name] > prob) for prob in probabilitiesToThreshold]
#                countsAboveProbabilityColumnNames = countsAboveProbabilityColumnNames + [class_name+' tile count (> ' + str(round(prob, 6)) + ')' for prob in probabilitiesToThreshold]
            #rows_list.append([os.path.split(case)[-1].replace('_inference.p', '')] + [np.count_nonzero(imPredictionImage > prob) for prob in probabilitiesToThreshold] + [np.count_nonzero(gastPredictionImage > prob) for prob in probabilitiesToThreshold] + [np.count_nonzero(respPredictionImage > prob) for prob in probabilitiesToThreshold] + [cytospongePathologistQCCall, cytospongePathologistCall, endoscopyC1M3Call, endoscopyC1M1Call, endoscopyC1Call, endoscopyC2Call, endoscopyC3Call])
#            rows_list.append([os.path.split(case)[-1].replace('_inference.p', '')] + countsAboveProbability + [atypiaGroundTruth, p53GroundTruth])
#            print(os.path.split(case)[-1].replace('_inference.p', ''))

#        else: # identify images with no matching ground truth or cases that appear in trainval set
#            cases_not_in_gt.append(caseId)

#    elif args.slidestouse == 'val':

    print("Case ID: "+caseID)
    #quit()

    # confirm that caseID is present in ground truth
    if caseID not in ground_truth['Case'].tolist():
        raise Warning(caseID+' not present in ground truth')

    # retrieve ground truth for caseID
    atypiaGroundTruth = int(ground_truth.loc[ground_truth['Case'] == caseID, 'Atypia-Pathologist'])
    atypia_positive_ground_truth_counter += atypiaGroundTruth
    #continue
    p53GroundTruth = int(ground_truth.loc[ground_truth['Case'] == caseID, 'P53-Pathologist'])
    p53_positive_ground_truth_counter += p53GroundTruth
    endoGroundTruth = ground_truth.loc[ground_truth['Case'] == caseID, 'Endoscopy-at-Cytosponge'].item()
    #print(endoGroundTruth)
    if endoGroundTruth not in ['CLE', 'NDBE', 'LGD', 'IND', 'HGD/IMC']:
        raise Warning('Endoscopy ground truth '+endoGroundTruth+' is not CLE, NDBE, LGD, IND, or HGD/IMC')

    print("Atypia ground truth: "+str(atypiaGroundTruth))
    print("P53 ground truth: "+str(p53GroundTruth))
    print("Endo ground truth: "+endoGroundTruth)


    #continue
    #print(class_names)
    tileDictionary = pickle.load(open(inferenceMapPath, "rb"))
    predictionImages = {class_name: np.zeros(tileDictionary['maskSize'][::-1]) for class_name in class_names}

    for key, val in tileDictionary['tileDictionary'].items():
        for class_idx, class_name in enumerate(class_names):
            predictionImages[class_name][key[1], key[0]] = val['prediction'][class_idx] if 'prediction' in val else 0

    ###print(caseId)

    #if caseId in best2GroundTruth['SampleID_2'].values:
    #'''
    #pre_correction_negcases = ["BEST2 CAM 0039", "BEST2 CAM 0092", "BEST2 CAM 0215", "BEST2 CAM 0351", "BEST2 CAM 0418", "BEST2 CAM 0470",
    #                "BEST2 CAM 0492", "BEST2 NEW 0004", "BEST2 NEW 0026", "BEST2 NEW 0071", "BEST2 NEW 0087", "BEST2 NEW 0110", "BEST2 NEW 0120",
    #                "BEST2 NOT 0013", "BEST2 NOT 0123", "BEST2 NOT 0143", "BEST2 NTE 0007", "BEST2 PBH 0001", "BEST2 POR 0009",
    #                "BEST2 STM 0016", "BEST2 STY 0009", "BEST2 UCL 0001", "BEST2 UCL 0055", "BEST2 UCL 0240", "BEST2 UCL 0286",
    #                "BEST2 CAM 0027", "BEST2 POR 0039", "BEST2 WEL 0014"] # changes
    #post_correction_negcases = ["BEST2 CAM 0039", "BEST2 CAM 0092", "BEST2 CAM 0215", "BEST2 CAM 0351", "BEST2 CAM 0418", "BEST2 CAM 0470",
    #                "BEST2 CAM 0492", "BEST2 NEW 0004", "BEST2 NEW 0026", "BEST2 NEW 0071", "BEST2 NEW 0087", "BEST2 NEW 0110", "BEST2 NEW 0120",
    #                "BEST2 NOT 0013", "BEST2 NOT 0123", "BEST2 NOT 0143", "BEST2 NTE 0007", "BEST2 PBH 0001", "BEST2 POR 0009",
    #                "BEST2 STM 0016", "BEST2 STY 0009", "BEST2 UCL 0001", "BEST2 UCL 0055", "BEST2 UCL 0240", "BEST2 UCL 0286",
    #                "BEST2 CAM 0127"] # changes
    #if caseID in post_correction_negcases:
    #    atypiaGroundTruth = 0
    #    p53GroundTruth = 0
    #else:
    #    atypiaGroundTruth = 1
    #''    p53GroundTruth = 0
    #'''
    #atypiaGroundTruth = 0
    #p53GroundTruth = 0




    #else:
    #    atypiaGroundTruth = int(best3GroundTruth.loc[(best3GroundTruth['BEST3_ID'] == caseId)]['Cytosponge_atypia']) #Cytosponge_atypia
    #    p53GroundTruth = int(best3GroundTruth.loc[(best3GroundTruth['BEST3_ID'] == caseId)]['Cytosponge_p53']) #Cytosponge_p53


    #quit()

    ###print(atypiaGroundTruth)
    ###continue

    countsAboveProbability = []
    countsAboveProbabilityColumnNames = []
    for class_name in class_names:
        countsAboveProbability = countsAboveProbability + [np.count_nonzero(predictionImages[class_name] > prob) for prob in probabilitiesToThreshold]
        countsAboveProbabilityColumnNames = countsAboveProbabilityColumnNames + [class_name+' tile count (> ' + str(round(prob, 7)) + ')' for prob in probabilitiesToThreshold]
    #rows_list.append([os.path.split(case)[-1].replace('_inference.p', '')] + [np.count_nonzero(imPredictionImage > prob) for prob in probabilitiesToThreshold] + [np.count_nonzero(gastPredictionImage > prob) for prob in probabilitiesToThreshold] + [np.count_nonzero(respPredictionImage > prob) for prob in probabilitiesToThreshold] + [cytospongePathologistQCCall, cytospongePathologistCall, endoscopyC1M3Call, endoscopyC1M1Call, endoscopyC1Call, endoscopyC2Call, endoscopyC3Call])
    rows_list.append([os.path.split(inferenceMapPath)[-1].replace('_inference.p', '')] + countsAboveProbability + [atypiaGroundTruth, p53GroundTruth, endoGroundTruth])
    #print(os.path.split(inferenceMapPath)[-1].replace('_inference.p', ''))


    # ranked tiles
    top100TileProbabilities = sorted(predictionImages[rankedTileClass].flatten(), reverse=True)[:100]
    rows_list_rankedtiles.append([os.path.split(inferenceMapPath)[-1].replace('_inference.p', '')] + top100TileProbabilities + [atypiaGroundTruth, p53GroundTruth, endoGroundTruth])





print('Number of ground truth atypia positive slides encountered:', atypia_positive_ground_truth_counter)
print('Number of ground truth aberrant P53 positive slides encountered:', p53_positive_ground_truth_counter)
hePredictions = pd.DataFrame(rows_list, columns=['Case'] + countsAboveProbabilityColumnNames + ['Pathologist atypia', 'Pathologist p53', 'Endoscopy'])
hePredictions.to_csv(os.path.join(outputPath, args.stain.upper()+'-prediction-data-'+os.path.basename(args.groundtruth)[:-4]+'-endo'+'.csv'), index=False) #'data/slideLevelAggregation/HE-data-' + whichArchitecture + '.csv', index=False)

# ranked tiles
top100TileProbabilitiesColumnNames = [str(u+1)+'_ranked_'+rankedTileClass+'_prob' for u in range(100)]
heRankedTileProbabilities = pd.DataFrame(rows_list_rankedtiles, columns=['Case'] + top100TileProbabilitiesColumnNames + ['Pathologist atypia', 'Pathologist p53', 'Endoscopy'])
heRankedTileProbabilities.to_csv(os.path.join(outputPath, args.stain.upper()+'-ranked-tile-probabilities-'+os.path.basename(args.groundtruth)[:-4]+'-endo'+'.csv'), index=False)
