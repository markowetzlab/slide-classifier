# Adam Berman

import argparse
import os
import pickle
import sys
import warnings
import glob
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generate t-sne.')
#parser.add_argument("-stain", required=True, help="he or p53")
parser.add_argument("-tsnedata", required=True, help="path to tsne_data.p file containing tnsne inference results to run tsne on")
parser.add_argument("-outputfolder", required=True, help="where to save tsne plot")
args = parser.parse_args()

stain = os.path.split(args.tsnedata)[-1].split('-')[1]
if stain == 'he':
    class_key = {0:'Artifact', 1:'Atypia', 2:'Background', 3:'Gastric cardia', 4:'Immune cells', 5:'Intestinal metaplasia', 6:'Respiratory mucosa', 7:'Squamous mucosa'}
    palette = {'Artifact':'#4C72B0', 'Atypia':'#DD8452', 'Background':'#55A868', 'Gastric cardia':'#C44E52', 'Immune cells':'#DA8BC3', 'Intestinal metaplasia':'#937860', 'Respiratory mucosa':'#8C8C8C', 'Squamous mucosa':'#8172B3'}
elif stain == 'p53':
    class_key = {0:'Aberrant P53 columnar epithelium', 1:'Artifact', 2:'Background', 3:'Immune cells', 4:'Squamous mucosa', 5:'Wild type columnar epithelium'}
    palette = {'Aberrant P53 columnar epithelium':'#FFCA2D', 'Artifact':'#4C72B0', 'Background':'#55A868', 'Immune cells':'#DA8BC3', 'Squamous mucosa':'#8172B3', 'Wild type columnar epithelium':'#C44E52'}
else: # #D4E600
    raise Warning('Stain must be he or p53')


# perform tsne
if os.path.isfile(args.tsnedata):
    results = pickle.load(open(args.tsnedata, 'rb'))
    #print(results)
    results_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(results['features'])
    #print(results_embedded.shape)
    results_embedded = pd.DataFrame(results_embedded, columns=['tsne_1', 'tsne_2'])
    results_embedded['tile_class'] = [class_key[num_key] for num_key in results['labels']]
    #print(results_embedded)
    #quit()

    sns.set(rc={'figure.figsize': (11,11)})
    p = sns.scatterplot(data=results_embedded, x='tsne_1', y='tsne_2', hue='tile_class', palette=palette)
    p.set(xlabel='t-SNE 1', ylabel='t-SNE 2')
    p.legend(title='Tile class')
    #p.set_size_inches(10,10)
    sns.color_palette('colorblind')
    #plt.figure(figsize=(10,10))
    plt.savefig(os.path.join(args.outputfolder, os.path.split(args.tsnedata)[-1].split('-tsneinference.py')[0]+'-tsneplot.png'), bbox_inches='tight')
    plt.show()

else:
    raise Warning(args.tsnedata+' is not a file.')
