import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from monai.data import CSVDataset, DataLoader
from monai.utils import first
from sklearn.manifold import TSNE
from slide_dataset import WSIDataset
import monai.transforms as mt

from torchvision import transforms
from torchvision import models

import pickle

arch='vit_small'
model_filename='/home/prew01/repos/dino-saur/HIPT/1-Hierarchical-Pretraining/ckpts/vits_40x_400_delta/vits_checkpoint.pth'
csv_filename='/media/prew01/Data/best/BEST2/atypia/annotated_tiles_lists/validation_annotated_tiles_lt-003.csv'
wsi_path='/media/prew01/Data/best/BEST2/atypia/test_slides'
atypia_only=False
resize=True
patch_size=256
file_format='.ndpi'
num_workers = 0
batch_size = 16
save_tmp_csv = False
plot_filename = 'results/delta_dino_40x_400'

def remove_module_from_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
 
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


print('read from csv:',csv_filename)
df = pd.read_csv(csv_filename)
if atypia_only:
    df = df[df['atypia_of_uncertain_significance'] == 1]
    csv_filename = csv_filename.replace('.csv','_atypia_only.csv')
    df.to_csv(csv_filename,index=False)
labels_list = df.columns[6:].values.tolist()
print('labels: ', labels_list)


preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(patch_size),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

datalist = CSVDataset(src=csv_filename,
    col_groups={
        "image":'FileName',
        "location":['y_','x_'], ##
        "label":[label for label in labels_list]
    },
    transform=mt.Lambdad("image", lambda x: os.path.join(wsi_path, x + file_format)))



for element in list(enumerate(datalist))[:5]:
    print(element)


dataset = WSIDataset(
            data=datalist, 
            patch_size=patch_size, 
            transform=preprocess,
            center_location=False,
            include_label=True,
            reader='openslide',
            resize=resize
            )


dataloader = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            )


first_sample = first(dataloader)
print(first_sample)


print("image: ")
print("    shape", first_sample["image"].shape) # batch_size, num_channels, tile_size, tile_size
print("    type: ", type(first_sample["image"]))
print("    dtype: ", first_sample["image"].dtype)
print("    image: ", first_sample["image"][0][:100])
print("labels: ")
print("    shape", first_sample["label"].shape) # batch_size, num_labels
print("    type: ", type(first_sample["label"]))
print("    dtype: ", first_sample["label"].dtype)
print("    labels: ", first_sample["label"])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

num_classes = len(labels_list)
unique_labels = [[float(i == j) for j in range(num_classes)] for i in range(num_classes)]

color_names = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']



print(plot_filename+'.npy')
if os.path.exists(plot_filename+'.npy'):
    print('saved model features already exist')
    tsne_features = np.load(plot_filename+'.npy')
    labels = np.load(plot_filename+'_labels.npy')
else:
    print('load model...')
    labels, features = [], []

    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    # model = vits.__dict__['vit_small'](patch_size=16, num_classes=0)

    pretrained_weights = model_filename
    # state_dict = torch.load(pretrained_weights, map_location=torch.device('cpu'))
    # state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/"+"dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth") #dino
    # state_dict = torch.load('/home/prew01/repos/dino-saur/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth', map_location=torch.device('cpu'))
    # checkpoint_key = 'teacher'
    # if checkpoint_key is not None and checkpoint_key in state_dict:
            # print(f"Take key {checkpoint_key} in provided checkpoint dict")
            # state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    # msg = model.load_state_dict(state_dict, strict=False)

    state_dict = torch.load('/home/prew01/repos/slide-classifier/models/trained_models/vit-l-16/he/he_vit_l_16.pt')
    model = models.vit_l_16(pretrained=False)
    model.load_state_dict(torch.load('/home/prew01/repos/slide-classifier/models/trained_models/vit-l-16/he/he_vit_l_16.pt').state_dict())
    # print('Pretrained weights found at {}\nLoaded with msg: {}'.format(pretrained_weights, msg))

    model.to(device)
    model.eval()

    print('extracting features...')
    
    for ind, batch in enumerate(dataloader):
        # print ind every 50 batches
        if ind % 50 == 0:
            print(f'batch: {ind}', end='\r')
        labels += batch['label']
        images = batch['image'].to(device)
        
        output = model.forward(images)

        current_outputs = output.cpu().detach().numpy().tolist()
        features = features + current_outputs

    print('example features:',features[:5])
    print('length of features:',len(features))

    labels = [i.numpy().tolist() for i in labels]
    print('example labels:',labels[:5])

    tsne_features = np.array(features)
    np_labels = np.array(labels)

    np.save(plot_filename, tsne_features)
    np.save(plot_filename+'_labels', np_labels)

tsne = TSNE(n_components=3, perplexity=5, random_state=42, learning_rate=200, n_iter=2000, metric='cosine').fit_transform(tsne_features)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# # extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]
tz = tsne[:, 2]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)
tz = scale_to_01_range(tz)

#Plot tsne features as different colours based on classification in 3D space
for ind, label in enumerate(unique_labels):
    if ind == 0 or ind == 2 or ind == 10:
        continue
    # if not (ind == 1 or ind ==8):
    #     continue
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(labels) if l == label]

    # extract the coordinates of the points of this class only
    current_tx = [tx[i] for i in indices]
    current_ty = [ty[i] for i in indices]
    current_tz = [tz[i] for i in indices]

    # convert the class color to matplotlib format
    color = color_names[ind % len(color_names)]

    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, current_tz, c=color, label=labels_list[ind])

# build a legend using the labels we set previously
ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

print(plot_filename)
plt.savefig(plot_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)

pickle.dump(fig, open(plot_filename+'.pkl', 'wb'))


