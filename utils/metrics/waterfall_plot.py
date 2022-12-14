# Adam Berman and Marcel Gehrung

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter, NullFormatter
from sklearn.metrics import classification_report, auc, confusion_matrix, roc_auc_score, roc_curve
from matplotlib import cm
import matplotlib.patches as patches
import pickle
import argparse
import os

#calibrationCohort = pickle.load( open( "../data/slideLevelAggregation/calibrationCohort.p", "rb" ) )
plt.style.use('figures/journal-style.mplstyle')
plt.rcParams["axes.labelweight"] = "bold"

parser = argparse.ArgumentParser(description='Make waterfall plot.')
parser.add_argument("-stain", required=True, help="he or p53")
parser.add_argument("-csvtoplot", required=True, help="Path to CSV file to plot")
#parser.add_argument("-runname", required=True, help="What to call the run for the figure title")
parser.add_argument("-thresholdtoplot", required=True, help="Threshold to plot")
args = parser.parse_args()

# go from 'BEST2 CAM 0158 R1 P53 - 2021-06-08 14.04.01' to 'BEST2/CAM/0158_R1'
def abbreviate_casenames(casename):
    # go from inference map path to this format: "BEST2/CAM/0001" or "BEST2/CAM/0001_R1"
    case_id = casename.split(' '+args.stain.upper()+' ')[0]
    caseIDSplit = case_id.split(' ')
    case_id = ''
    numPieces = len(caseIDSplit)
    for i, casePiece in enumerate(caseIDSplit):
        if 'P53' in casePiece or 'HE' in casePiece:
            break
        if i == 0:
            case_id = casePiece
        elif 'R' in casePiece and i == (numPieces-1):
            case_id = case_id + '_' + casePiece
        else:
            case_id = case_id + '/' + casePiece
    return case_id

if args.stain not in ['he', 'p53']:
    raise Warning("-stain argument must be either 'he' or 'p53'")

if args.stain == 'he':
    columnGTName = 'atypia'
    columnWord = 'atypia'
else:
    columnGTName = 'p53'
    columnWord = 'aberrant_positive_columnar'

def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return z

def remove_spines(axes):
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

#qcScore = pd.read_csv('data/slideLevelAggregation/HE-data-vgg16.csv')

#qcScore['Case'] = qcScore['Case'].str.replace('_HE_1', '')
#qcScoreColumn = 'Gastric count (> 0.99)'
#imQcScoreColumn = 'IM positive count (> 0.995)'

#qcScore['dummyColumn'] = 0.3
diagScore = pd.read_csv(args.csvtoplot) #'data/slideLevelAggregationTestSet/HE-prediction-data.csv'
diagScore['dummyColumn'] = 0.3

#diagScore['Case'] = diagScore['Case'].str.replace('_TFF3_1', '')
diagScoreColumn = columnWord+' tile count (> '+str(args.thresholdtoplot)+')' #'atypia tile count (> 0.999)'

#mergedScores = pd.merge(qcScore[['Case',qcScoreColumn,imQcScoreColumn,'Pathologist atypia','dummyColumn']],diagScore[['Case',diagScoreColumn,'Endoscopy (at least C1M3)','Cytosponge']],on="Case")

#mergedScoresCalibration = mergedScores[mergedScores['Case'].isin(calibrationCohort)]
#print(len(mergedScoresCalibration))

#diagTileCounts, caseLabels, positiveTileCounts, heImTileCounts, qcGroundTruth, groundTruth, pathCallCytosponge =
#####atypiaTileCounts, caseLabels, atypiaGroundTruth = zip(*sorted(zip(diagScore[diagScoreColumn].tolist(), diagScore['Case'].tolist(), diagScore['Pathologist atypia'].tolist()),reverse=True))
endo_label_ranker = {'CLE': 'V_CLE', 'NDBE': 'W_NDBE', 'IND': 'X_IND',
                    'LGD': 'Y_LGD', 'HGD/IMC': 'Z_HGD/IMC'}
sortableEndoGroundTruth = [endo_label_ranker[endo_label] for endo_label in diagScore['Endoscopy'].tolist()]
atypiaTileCounts, atypiaGroundTruth, endoGroundTruth, caseLabels = zip(*sorted(zip(diagScore[diagScoreColumn].tolist(), diagScore['Pathologist '+columnGTName].tolist(), sortableEndoGroundTruth, diagScore['Case'].tolist()),reverse=True))
#print(atypiaTileCounts, caseLabels, atypiaGroundTruth)
#quit()


caseLabels = list(caseLabels)
atypiaTileCounts = list(atypiaTileCounts)
#groundTruth = list(groundTruth)
#qcGroundTruth = list(qcGroundTruth)
#positiveTileCounts = list(positiveTileCounts)
atypiaGroundTruth = list(atypiaGroundTruth)
#heImTileCounts = list(heImTileCounts)
endoGroundTruth = list(endoGroundTruth)


#####atypiaTileCounts, atypiaGroundTruth = zip(*sorted(zip(atypiaTileCounts, atypiaGroundTruth),reverse=True))
atypiaTileCounts, atypiaGroundTruth, endoGroundTruth, caseLabels = zip(*sorted(zip(atypiaTileCounts, atypiaGroundTruth, endoGroundTruth, caseLabels),reverse=True))

#print(atypiaTileCounts)
#print(atypiaGroundTruth)
#print(caseLabels)
#quit()

#diagColormap = [[30/255, 120/255, 182/255],[232/255, 23/255, 30/255]]
if args.stain == 'he':
    atypiaColormap = [[1,1,1],[200/255,97/255,161/255]] #[[1,1,1],[.45,.45,.45]] # these are the colors of the upper bar, first color is GT negative, second color is GT positive
else:
    atypiaColormap = [[1,1,1],[140/255,82/255,60/255]]

endoColormap = {'V_CLE': [1,1,1], 'W_NDBE': [1,1,1], 'X_IND': [254/255,153/255,0/255],
                    'Y_LGD': [200/255,0/255,0/255], 'Z_HGD/IMC': [127.5/255,0/255,0/255]}
#endoColormap = {'CLE': [1,1,1], 'NDBE': [1,1,1], 'IND': [254/255,153/255,0/255],
#                    'LGD': [200/255,0/255,0/255], 'HGD/IMC': [127.5/255,0/255,0/255]}
#{'CLE': [1,1,1], 'NDBE': [1,1,1], 'IND': [254/255,255/255,0/255],
#                    'LGD': [254/255,153/255,0/255], 'HGD/IMC': [254/255,0/255,0/255]}

#pathCytoColormap = [[1,1,1],[231/255,134/255,133/255]]

fig = plt.figure(figsize=(10.5,3))

gs = gridspec.GridSpec(nrows=3,
                       ncols=1,
                       figure=fig,
                       height_ratios=[0.14, 0.14, 0.72],#[0.17,0.83],
                       #width_ratios= [0.04, 0.41, 0.1, 0.04, 0.42],
                       hspace=0.18)

ax2 = fig.add_subplot(gs[0, 0])
#ax1.scatter(np.ones((len(mergedScores),1)),np.arange(0,len(mergedScores)),color=[qcColormap[gt] for gt in groundTruth],s=2)
ax2.bar(caseLabels,diagScore['dummyColumn'],color=[endoColormap[gt] for gt in endoGroundTruth],width=1.1)

ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.set_xticklabels(['' for tick in ax2.get_yticks()])
ax2.set_xlabel('Patients', fontsize=13)
ax2.set_ylabel('Endoscopy',rotation=0, fontsize=11)
ax2.yaxis.set_label_coords(-0.11,0.05)
ax2.xaxis.set_label_coords(0.5,1.5)
#ax0.yaxis.set_label_position("left")
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top')
ax2.get_yaxis().set_ticks([])
ax2.get_xaxis().set_ticks([])
ax2.set_xlim(0,len(diagScore))
ax2.set_ylim(0,0.3)


ax0 = fig.add_subplot(gs[1, 0])
#ax1.scatter(np.ones((len(mergedScores),1)),np.arange(0,len(mergedScores)),color=[qcColormap[gt] for gt in groundTruth],s=2)
ax0.bar(caseLabels,diagScore['dummyColumn'],color=[atypiaColormap[gt] for gt in atypiaGroundTruth],width=1.1)

ax0.spines['left'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.set_xticklabels(['' for tick in ax0.get_yticks()])
#ax0.set_xlabel('Patients', fontsize=13)
if args.stain == 'he':
    ax0.set_ylabel('Pathologist slide-level\natypia ground truth',rotation=0, fontsize=11)
else:
    ax0.set_ylabel('Pathologist slide-level\nP53 ground truth',rotation=0, fontsize=11)
ax0.yaxis.set_label_coords(-0.11,0.05)
ax0.xaxis.set_label_coords(0.5,1.5)
#ax0.yaxis.set_label_position("left")
ax0.xaxis.tick_top()
ax0.xaxis.set_label_position('top')
ax0.get_yaxis().set_ticks([])
ax0.get_xaxis().set_ticks([])
ax0.set_xlim(0,len(diagScore))
ax0.set_ylim(0,0.3)





ax1 = fig.add_subplot(gs[2, 0])

viridis = cm.get_cmap('Greens', 101)
fancyColormap = [viridis(x) for x in np.arange(0,10001)]
if args.stain == 'he':
    ax1.plot(caseLabels,atypiaTileCounts,color=[200/255,97/255,161/255])#[0.525, 0.137, 0.373])
    plt.fill_between(caseLabels,atypiaTileCounts, 0,
                     facecolor=[1.000, 0.851, 0.949], # The fill color
                     color=[235/255,204/255,220/255],#[1.000, 0.851, 0.949],       # The outline color
                     alpha=1)          # Transparency of the fill
else:
    ax1.plot(caseLabels,atypiaTileCounts,color=[140/255,82/255,60/255])#[0.525, 0.137, 0.373])#[0.825, 0.537, 0.173])#[0.525, 0.137, 0.373])
    plt.fill_between(caseLabels,atypiaTileCounts, 0,
                     facecolor=[1.000, 0.851, 0.949], # The fill color
                     color=[231/255,221/255,218/255], #[1.000, 0.851, 0.949],       # The outline color
                     alpha=1)          # Transparency of the fill

ax1.set_yscale('log')
ax1.yaxis.tick_left()
ax1.yaxis.set_label_position("left")
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_xticklabels(['' for tick in ax1.get_yticks()])
if args.stain == 'he':
    ax1.set_ylabel('Number of\n'+columnGTName+' tiles\n detected by model\n(threshold='+str(args.thresholdtoplot)+')', fontsize=11)
else:
    ax1.set_ylabel('Number of\nP53 aberrant tiles\n detected by model\n(threshold='+str(args.thresholdtoplot)+')', fontsize=11)
ax1.yaxis.set_label_position("left")
ax1.yaxis.set_tick_params(labelsize=12)
ax1.get_xaxis().set_visible(False)
ax1.set_xlim(0,len(diagScore))
ax1.set_ylim(0,1000)
ax1.yaxis.set_major_formatter(ScalarFormatter())
ax1.yaxis.set_minor_formatter(NullFormatter())
ax1.set_yticks([1,10,100,1000])
ax1.set_yticklabels(['1','10','100','1000'])
#ax1.axvline(x=137,ymin=0,ymax=1000,lw=3,color=[1,215/255,165/255,.9])
#ax1.axvline(x=164,ymin=0,ymax=1000,lw=3,color=[1,158/255,166/255,.9])

viridis = cm.get_cmap('Oranges', 101)
fancyColormap = [viridis(x) for x in np.arange(0,101)]

plt.show()

#plt.savefig('figures/'+args.runname+'-horizontal-calibration.pdf', bbox_inches='tight') #'figures/atypiaPlot-horizontal-calibration.pdf'
plt.savefig(os.path.join(os.path.split(args.csvtoplot)[0], 'horizontalCalibration'+args.stain.upper()+'.pdf'), bbox_inches='tight')


df = pd.DataFrame(list(zip([abbreviate_casenames(calab) for calab in caseLabels], atypiaGroundTruth, atypiaTileCounts, [egt[2:] for egt in endoGroundTruth])),
               columns =['case', columnGTName+'_ground_truth', columnGTName+'_tile_count', 'endoscopy'])
df_sorted = df.sort_values(by=['case'])

df_name_core = args.csvtoplot.split('-prediction-data-just_labels')[1].split('-endo.csv')[0]
if '/val/' in args.csvtoplot:
    df_name = columnGTName+'_val_'+df_name_core+'_results.csv'
elif '/test/' in args.csvtoplot:
    df_name = columnGTName+'_test_'+df_name_core+'_results.csv'
else:
    raise Warning('Could not detect val or test set detail from -csvtoplot')
print(os.path.join(os.path.split(args.csvtoplot)[0], df_name))
df_sorted.to_csv(os.path.join(os.path.split(args.csvtoplot)[0], df_name), index=False)#'tileCountGroundTruth.csv'))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)
