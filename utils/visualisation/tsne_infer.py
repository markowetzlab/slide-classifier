# Adam Berman

from PIL import Image
import numpy as np
import torch
import argparse
import os
import pickle
from PIL import Image
from torchvision import models, transforms
import sys
import warnings
import torch.nn as nn
from torch.autograd import Variable
#import imageio
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
#import pickle
#import os
#import argparse
from pathlib import Path
from dataset.EmptyDirAcceptingImageFolder import EmptyDirAcceptingImageFolder
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

parser = argparse.ArgumentParser(description='Generate t-sne.')
#parser.add_argument("-stain", required=True, help="he or p53")
parser.add_argument("-model", required=True, help="which model to infer gradcam on") #/media/berman01/Seagate Ext/atypia_p53_tsne/partSize-he-106-27-vgg19_bn_19-5-2022-epoch-39_ft.pt
parser.add_argument("-tilefolder", required=True, help="path to folder containing case folders containing class folders containing tiles")
parser.add_argument("-whichcase", required=True, help="which case within tilefolder to generate tsne from the tiles of; if a trainval_split.p file is included, will be run on all validation cases")
parser.add_argument("-outputfolder", required=True, help="where to output tiles")
parser.add_argument("-batchsize", default=1, type=int, help="batch size (default is 1)")
#parser.add_argument("-whichclasses", default=None, help="which classes to extract gradcams for, where classes are separated by commas; default is all classes")
args = parser.parse_args()


#if args.stain == 'he':
#    class_key = {0:'Artifact', 1:'Atypia', 2:'Background', 3:'Gastric cardia', 4:'Immune cells', 5:'Intestinal metaplasia', 6:'Respiratory mucosa', 7:'Squamous mucosa'}
#elif args.stain == 'p53':
#    class_key = {0:'Aberrant P53 columnar epithelium', 1:'Artifact', 3:'Background', 4:'Immune cells', 5:'Squamous mucosa', 6:'Wild type columnar epithelium'}


# perform tsne

'''
if os.path.isfile('he_raw_tsne_data.p'):
    results = pickle.load(open('he_raw_tsne_data.p', 'rb'))
    #print(results)
    results_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(results['features'])
    #print(results_embedded.shape)
    results_embedded = pd.DataFrame(results_embedded, columns=['tsne_1', 'tsne_2'])
    results_embedded['tile_class'] = [class_key[num_key] for num_key in results['labels']]
    #print(results_embedded)
    #quit()

    p = sns.scatterplot(data=results_embedded, x='tsne_1', y='tsne_2', hue='tile_class')
    p.set(xlabel='t-SNE 1', ylabel='t-SNE 2')
    p.legend(title='Tile class')
    sns.color_palette('colorblind')
    plt.show()

    quit()
'''


#if args.whichclasses:
#    which_classes = args.whichclasses.split(',')
if args.whichcase[-2:] == '.p':
    trainval_split = pickle.load(open(args.whichcase, 'rb'))
    val_cases = []
    for val_case in trainval_split['val_cases']:
        val_cases.append(os.path.split(val_case)[-1].replace('.ndpi', ''))
    val_cases.sort()
else:
    val_cases = [args.whichcase]#os.path.split(args.tilefolder)[-1]]
#print('val_cases:', val_cases)


#from WholeSlideImageDataset import WholeSlideImageDataset

#sys.path.append('/home/cri.camres.org/berman01/pathml') # install from here: https://github.com/9xg/pathml
#from pathml import slide

architecture = os.path.split(args.model)[-1].split('-')[4]
if not os.path.exists(args.outputfolder):
    os.makedirs(args.outputfolder, exist_ok=True)
#print('Output location:', os.path.join(args.outputfolder, os.path.split(args.model)[-1].split('_ft.pt')[0]+'-tsneinference.p'))
stain = os.path.split(args.model)[-1].split('-')[1]
if stain == 'he':
    classes = ['artifact', 'atypia', 'background', 'gastric_cardia', 'immune_cells',
                    'intestinal_metaplasia', 'respiratory_mucosa', 'squamous_mucosa']
elif stain == 'p53':
    classes = ['aberrant_positive_columnar', 'artifact', 'background', 'immune_cells',
                    'squamous_mucosa', 'wild_type_columnar']
else:
    raise Warning('Stain must be he or p53')
#print('Architecture:', architecture)
#print('Stain:', stain)
#print('Epoch:', )

#if args.architecture == "resnet18":
#    trainedModel = models.resnet18(pretrained=True)
#    trainedModel.fc = nn.Linear(512, int(args.numclasses))
#elif args.architecture == "inceptionv3":
#    trainedModel = models.inception_v3(pretrained=True)
#    num_ftrs = trainedModel.fc.in_features
#    trainedModel.AuxLogits.fc = nn.Linear(768, int(args.numclasses))
#    trainedModel.fc = nn.Linear(num_ftrs, int(args.numclasses))
#elif args.architecture == "vgg16":
#    trainedModel = models.vgg16(pretrained=True)
#    trainedModel.classifier[6] = nn.Linear(4096, int(args.numclasses))
#elif args.architecture == "vgg19_bn":
if architecture == "vgg19_bn":
    trainedModel = models.vgg19_bn(pretrained=True)
    trainedModel.classifier[6] = nn.Linear(4096, len(classes))
    trainedModel.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    trainedModel.classifier = trainedModel.classifier[:-1]
    print(trainedModel)

elif architecture == 'vit_l_16':
    trainedModel = models.vit_l_16(pretrained=True)
    #print(trainedModel)
    #quit()
    trainedModel.heads[-1] = nn.Linear(1024, len(classes))
    trainedModel.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    trainedModel.heads[-1] = nn.Identity()
    #trainedModel.heads[-1] = nn.Linear(1024, 1000)
    #trainedModel.heads = trainedModel.heads[-1]
    #trainedModel.heads = trainedModel.heads[-1]
    print(trainedModel)

elif architecture == 'resnext101_32x8d':
    trainedModel = models.resnext101_32x8d(pretrained=True)
    trainedModel.fc = nn.Linear(trainedModel.fc.in_features, len(classes))
    trainedModel.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    trainedModel.fc = nn.Identity()

    print(trainedModel)
    #quit()

elif architecture == 'convnext_large':
    trainedModel = models.convnext_large(pretrained=True)
    trainedModel.classifier[-1] = nn.Linear(1536, len(classes))
    trainedModel.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    trainedModel.classifier[-1] = nn.Identity()

    print(trainedModel)

#elif args.architecture == "densenet":
#    trainedModel = models.densenet121(pretrained=True)
#    trainedModel.classifier = nn.Linear(1024, int(args.numclasses))
#elif args.architecture == "alexnet":
#    trainedModel = models.alexnet(pretrained=True)
#    trainedModel.classifier[6] = nn.Linear(4096, int(args.numclasses))
#elif args.architecture == "squeezenet":
#    trainedModel = models.squeezenet1_1(pretrained=True)
#    trainedModel.classifier[1] = nn.Conv2d(
#        512, int(args.numclasses), kernel_size=(1, 1), stride=(1, 1))
#    trainedModel.num_classes = int(args.numclasses)
else:
    raise Warning(architecture+' is not recognized.')

#quit()





#quit()
#whichStain = args.stain
pathTileSize = 400
batchSizeForInference = args.batchsize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Inferring on GPU")
else:
    print("Inferring on CPU")
trainedModel.to(device)
trainedModel.eval()

#if args.architecture == 'vgg16' or args.architecture == 'vgg19_bn' or args.architecture == 'resnet18' or args.architecture == 'squeezenet' or args.architecture == 'densenet' or args.architecture == "alexnet":
if architecture != 'inception_v3':
    patch_size = 224
else:
#if args.architecture == 'inceptionv3':
    patch_size = 299

#else:
if os.path.exists(os.path.join(args.tilefolder, 'channel_means_and_stds.p')):
    channel_means_and_stds = pickle.load(open(os.path.join(args.tilefolder, 'channel_means_and_stds.p'), 'rb'))
    channel_means = channel_means_and_stds['channel_means']
    channel_stds = channel_means_and_stds['channel_stds']
else:
    raise Warning('channel_means_and_stds.p must be present in tilefolder')

print('channel means:', channel_means)
print('channel_stds:', channel_stds)

data_transforms = transforms.Compose([
    transforms.Resize(patch_size),
    transforms.ToTensor(),
    transforms.Normalize(channel_means, channel_stds)
])


tile_dataset = torch.utils.data.ConcatDataset([EmptyDirAcceptingImageFolder(os.path.join(
    args.tilefolder, val_case), data_transforms) for val_case in val_cases])


print(len(tile_dataset))
#quit()

if architecture == 'vgg19_bn':
    results = np.empty((0,4096))
elif architecture == 'vit_l_16':
    results = np.empty((0,1024))
elif architecture == 'resnext101_32x8d':
    results = np.empty((0,2048))
elif architecture == 'convnext_large':
    results = np.empty((0,1536))
else:
    raise Warning(architecture+' not supported for feature count.')

result_labels = []

dataloader = torch.utils.data.DataLoader(tile_dataset, batch_size=16, num_workers=8)

for inputs, labels in tqdm(dataloader):
    inputTile = inputs.to(device)
    output = trainedModel(inputTile)
    #output = output.to(device)
    #print('output.detach().np():', output.detach().numpy())
    #print('shape:', output.detach().numpy().shape)

    results = np.append(results, output.detach().numpy(), axis=0)

    #print('results:', results)
    print('results shape:', results.shape)


    #batch_prediction = torch.nn.functional.softmax(
    #    output, dim=1).cpu().data.numpy()
    result_labels = result_labels + labels.tolist()
    #print('labels:', result_labels)
    #print('batch_prediction:', batch_prediction)

    #break

pickle.dump({'features': results, 'labels': result_labels}, open(os.path.join(args.outputfolder, os.path.split(args.model)[-1].split('_ft.pt')[0]+'-tsneinference.p'), 'wb'))

quit()

    # Reshape it is a Todo - instead of for looping
#    for index in range(len(inputTile)):
#        tileAddress = (inputs['tileAddress'][0][index].item(),
#                       inputs['tileAddress'][1][index].item())
#        pathSlide.appendTag(tileAddress, 'prediction', batch_prediction[index, ...])

#for inputs, labels in tqdm(dataloader):
#    inputs = inputs.to(device)
#    labels = labels.to(device)
#    _, preds = torch.max(outputs, 1)
















'''
for caseName in tqdm(cases):

    print(caseName)

    if os.path.isfile(os.path.join(inferenceMapFolder, caseName + '_inference.p')):
        print("Case already processed. Skipping...")
        continue

    #pathSlide = slide.Slide(caseFolder)
    pathSlide = slide.Slide(os.path.join(slidesRootFolder, caseName + '.' + args.wsiextension))
    pathSlide.setTileProperties(tileSize=pathTileSize, tileOverlap=float(args.tileoverlap))

    if args.foregroundonly in ['True', 'true', 'TRUE']:
        # , tileOverlap=0.33
        pathSlide.detectForeground(threshold=95)

        pathSlideDataset = WholeSlideImageDataset(
            pathSlide, foregroundOnly=True, transform=data_transforms)#, foregroundOnly=True)
    else:
        pathSlideDataset = WholeSlideImageDataset(
            pathSlide, foregroundOnly=False, transform=data_transforms)

    since = time.time()
    pathSlideDataloader = torch.utils.data.DataLoader(pathSlideDataset, batch_size=batchSizeForInference, shuffle=False, num_workers=16)
    for inputs in tqdm(pathSlideDataloader):
        inputTile = inputs['image'].to(device)
        output = trainedModel(inputTile)
        output = output.to(device)

        batch_prediction = torch.nn.functional.softmax(
            output, dim=1).cpu().data.numpy()

        # Reshape it is a Todo - instead of for looping
        for index in range(len(inputTile)):
            tileAddress = (inputs['tileAddress'][0][index].item(),
                           inputs['tileAddress'][1][index].item())
            pathSlide.appendTag(tileAddress, 'prediction', batch_prediction[index, ...])
    tileDictionaryWithInference = {'maskSize': (
        pathSlide.numTilesInX, pathSlide.numTilesInY), 'tileDictionary': pathSlide.tileDictionary}#pathSlide.tileMetadata}
    pickle.dump(tileDictionaryWithInference, open(
        os.path.join(inferenceMapFolder, caseName + '_inference.p'), 'wb'))
    time_elapsed = time.time() - since
    print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
'''















#-------------------------------------------------------------------------------





#model = models.vgg19_bn(pretrained=True)
#model.classifier[6] = torch.nn.Linear(4096, int(8))
#model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

#epoch = 800
#PATH = 'vgg16_epoch{}.pth'.format(epoch)
#checkpoint = torch.load(PATH)
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch = checkpoint['epoch']


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # Mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var



class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

model.classifier._modules['6'] = Identity()
model.eval()
logits_list = np.empty((0,4096))
targets = []


with torch.no_grad():
    for step, (t_image, target, classess, image_path) in enumerate(test_loader):

        t_image = t_image.cuda()
        target = target.cuda()
        target = target.data.cpu().np()
        targets.append(target)

        logits = model(t_image)
        print(logits.shape)

        logits = logits.data.cpu().np()
        print(logits.shape)
        logits_list = np.append(logits_list, logits, axis=0)
        print(logits_list.shape)


tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
tsne_results = tsne.fit_transform(logits_list)

target_ids = range(len(targets))

plt.scatter(tsne_results[:,0],tsne_results[:,1],c = target_ids ,cmap=plt.cm.get_cmap("jet", 14))
plt.colorbar(ticks=range(14))
plt.legend()
plt.show()
