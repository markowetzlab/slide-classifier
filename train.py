import argparse
import copy
import datetime
import glob
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
							 precision_score, recall_score)
from torch.optim import lr_scheduler
from tqdm import tqdm

from dataset_processing.best import EmptyDirAcceptingImageFolder, ImbalancedDatasetSampler
from models import get_network
from dataset_processing import class_parser
from dataset_processing.image import channel_averages, transforms

# This file enables training of different deep learning architectures based on the extracted tiles

def parse_args():
	parser = argparse.ArgumentParser(description='Run training on tiles.')

	#essentials
	parser.add_argument("--stain", type=str, required=True, help="he or p53")
	parser.add_argument("--network", type= str, required=True, help="which CNN network to use (vgg11_bn, vgg16, vgg19_bn, resnet18, resnet152, googlenet, inception_v3, squeezenet1_1, densenet121, densenet161, alexnet, resnext101_32x8d, shufflenet_v2_x1_0, mobilenet_v3_large, wide_resnet101_2, mnasnet1_0, 'efficientnet_b0, 'efficientnet_b7, regnet_x_32gf, regnet_y_32gf, vit_b_16, vit_l_16, convnext_tiny, convnext_large)")
	parser.add_argument("--tile_path", type=str, required=True, help="tile folder")
	parser.add_argument("--output", type=str, default='results', help="output folder")
	parser.add_argument("--trainingfraction", type=float, default=0.80, help="fraction of patients used for training instead of validation")

	#class variables
	parser.add_argument("--dysplasia_separate", action='store_false', help="Flag whether to separate the atypia of uncertain significance and dysplasia classes")
	parser.add_argument("--respiratory_separate", action='store_false', help="Flag whether to separate the respiratory mucosa cilia and respiratory mucosa classes")
	parser.add_argument("--gastric_separate", action='store_false', help="Flag whether to separate the tickled up columnar and gastric cardia classes")
	parser.add_argument("--atypia_separate", action='store_false', help="Flag whether to perform the following class split: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other")
	parser.add_argument("--p53_separate", action='store_false', help="Flag whether to perform the following class split: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar")
	
	#eptimisers
	parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs to train the model for")
	parser.add_argument("--lr", type=float, default=0.001, help="learning rate used for training")
	parser.add_argument("--lr_momentum", type=float, default=0.9, help="lr_momentum used for training")
	parser.add_argument("--lr_step_size", type=int, default=7, help="number of epochs between learning rate steps of a factor of lr_gamma")
	parser.add_argument("--lr_gamma", type=float, default=0.1, help="factor by which to reduce learning rate every lr_step_size epochs")

	#training variables
	parser.add_argument("--seed", default=366, type=int, help="seed used to partition training and validation cases")
	parser.add_argument("--channelmeans", default=None, help="0-1 normalized colour channel means for all tiles on dataset separated by commas, e.g. 0.485,0.456,0.406 for RGB, respectively. If not speficied, these means will be taken from 'channel_means_and_stds.p' in the -tile_path")
	parser.add_argument("--channelstds", default=None, help="0-1 normalized colour channel standard deviations for all tiles on dataset separated by commas, e.g. 0.229,0.224,0.225 for RGB, respectively. If not speficied, these standard deviations will be taken from 'channel_means_and_stds.p' in the -tile_path")
	parser.add_argument("--colourjitter", type=float, default=0.0, help="how much to jitter (brightness, contrast, saturation, hue), should be 0-1")
	parser.add_argument("--useimbalanceddatasetsampler", action="store_true", help="whether to use ImbalancedDatasetSampler")
	
	parser.add_argument('--silent', action='store_true', help='Flag which silences tqdm on servers')

	args = parser.parse_args()

	if args.stain not in ['he', 'p53', 'tff3']:
		raise Warning("-stain argument must be either 'he' or 'p53' or 'tff3'")

	return args

def train_model(model, params, device, criterion, optimizer, scheduler, dataloaders, num_epochs=25, silent=False):
	best_model_wts = copy.deepcopy(model.state_dict())
	best_epoch = 0
	best_acc = 0.0

	class_names = params['classes']

	for epoch in range(num_epochs):
		print(f'Epoch {epoch}/{num_epochs - 1}')
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			epoch_ground_truth = []
			epoch_predictions = []
			running_loss = 0.0
			running_corrects = 0
			running_tp = {class_name: 0 for class_idx, class_name in enumerate(class_names)}
			running_fp = {class_name: 0 for class_idx, class_name in enumerate(class_names)}
			running_tn = {class_name: 0 for class_idx, class_name in enumerate(class_names)}
			running_fn = {class_name: 0 for class_idx, class_name in enumerate(class_names)}

			# Iterate over data.
			for inputs, labels in tqdm(dataloaders[phase], disable=silent):
				inputs = inputs.to(device)
				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					if params['network_name'] == "inception_v3" and phase == 'train':
						outputs, aux = model(inputs)
					else:
						outputs = model(inputs)
					_, preds = torch.max(outputs, 1)

					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				epoch_ground_truth = epoch_ground_truth + labels.data.tolist()
				epoch_predictions = epoch_predictions + preds.tolist()

				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
				running_tp = {class_name: running_tp[class_name] + torch.sum((preds == class_idx) & (labels.data == class_idx)) for class_idx, class_name in enumerate(class_names)}
				running_fp = {class_name: running_fp[class_name] + torch.sum(((preds == class_idx).long() - (labels.data == class_idx).long()) == 1) for class_idx, class_name in enumerate(class_names)}
				running_tn = {class_name: running_tn[class_name] + torch.sum(((preds == class_idx).long() + (labels.data == class_idx).long()) == 0) for class_idx, class_name in enumerate(class_names)}
				running_fn = {class_name: running_fn[class_name] + torch.sum(((preds == class_idx).long() - (labels.data == class_idx).long()) == -1) for class_idx, class_name in enumerate(class_names)}

			if phase == 'train':
				scheduler.step()

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			epoch_acc = accuracy_score(epoch_ground_truth, epoch_predictions)
			epoch_weighted_acc = balanced_accuracy_score(epoch_ground_truth, epoch_predictions)  # accuracy accounting for class imbalance
			epoch_weighted_rec = recall_score(epoch_ground_truth, epoch_predictions, average='weighted')  # average recall accounting for class imbalance
			epoch_weighted_prec = precision_score(epoch_ground_truth, epoch_predictions, average='weighted')  # average precision accounting for class imbalance
			epoch_weighted_f1 = f1_score(epoch_ground_truth, epoch_predictions, average='weighted')  # average F1 score accounting for class imbalance

			epoch_rec = {class_name: ((running_tp[class_name].item() / (running_tp[class_name].item() + running_fn[class_name].item())) if (running_tp[class_name].item() != 0) else 0) for class_idx,
						 class_name in enumerate(class_names)}
			epoch_prec = {class_name: ((running_tp[class_name].item() / (running_tp[class_name].item() + running_fp[class_name].item())) if (running_tp[class_name].item() != 0) else 0) for class_idx,
						  class_name in enumerate(class_names)}

			#need to change output
			stats[phase].append(
				{'loss': epoch_loss, 'accuracy': epoch_acc, 'precision': epoch_prec, 'recall': epoch_rec,
					'weighted_accuracy': epoch_weighted_acc,
					'weighted_precision': epoch_weighted_prec,
					'weighted_recall': epoch_weighted_rec,
					'weighted_f1': epoch_weighted_f1})

			print(f'Phase Loss: {phase:.4f} Acc: {epoch_loss:.4f} Weighted Acc: {epoch_weighted_acc:.4f} Weighted Pre: {epoch_weighted_prec:.4f}')
			print(f'Weighted Rec: {epoch_weighted_rec:.4f} Weighted F1: {epoch_weighted_f1:.4f} Rec: {epoch_rec:.4f} Pre: {epoch_prec:.4f}')

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_epoch = epoch
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

			if phase == 'val':
				torch.save(model.state_dict(), os.path.join(args.output, 'trained_models', 'partSize-' + stain + '-' +
					str(len(train_cases)) + '-' + str(len(val_cases)) + '-' + str(network) + '_' +
					str(now.day) + '-' + str(now.month) + '-' + str(now.year) + '-epoch-' + str(epoch) + '_ft.pt'))

		print('Best epoch by val acc: '+str(best_epoch))
		print('Best val acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model, params

if __name__ == '__main__':
	args = parse_args()
	
	network = args.network
	stain = args.stain
	data_dir = args.tile_path
	data_path = glob.glob(os.path.join(data_dir, "BEST*"))

	now = datetime.datetime.now()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		print("Training on GPU")
	else:
		print("Training on CPU")

	random.seed(args.seed)

	random.shuffle(data_path)
	cases = [os.path.split(case)[-1] for case in data_path]
	val_size = np.floor(len(cases)/int(args.folds))

	num_train_cases = round(args.trainingfraction * len(cases))
	num_val_cases = len(cases) - num_train_cases
	training_partitions = [num_train_cases]
	val_cases = cases[0:num_val_cases]

	if not os.path.exists(os.path.join(args.output, 'trained_models')):
		os.makedirs(os.path.join(args.output, 'trained_models'))

	# Save train and val case split
	trainval_split = {'train_cases': data_path[-1 - num_train_cases:-1], 'val_cases': data_path[0:num_val_cases]}
	pickle.dump(trainval_split, open(os.path.join(args.output, "trainval_split.pickle"), 'wb'))

	class_names = class_parser(args.stain, args.dysplasia_separate, args.respiratory_separate, args.gastric_separate, args.atypia_separate, args.p53_separate)

	class_count = [0] * len(class_names)
	class_count_val = [0] * len(class_names)

	for caseName in cases[-1 - num_train_cases:-1]:
		for class_idx, class_name in enumerate(class_names):
			path, dirs, files = next(
				os.walk(os.path.join(data_dir, caseName, class_name)))
			file_count = len(files)
			class_count[class_idx] += file_count
	print('Class names: ', class_names)
	print('Sample count for training dataset: ', class_count)

	for caseName in val_cases:
		for class_idx, class_name in enumerate(class_names):
			path, dirs, files = next(
				os.walk(os.path.join(data_dir, caseName, class_name)))
			file_count = len(files)
			class_count_val[class_idx] += file_count
	print('Sample count for validation dataset: ', class_count_val)

	channel_means, channel_stds = channel_averages(args.tile_path, args.channelmeans, args.channelstds)

	print('channel means:', channel_means)
	print('channel_stds:', channel_stds)

	model_ft, params = get_network(args.network, class_names)

	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model_ft = nn.DataParallel(model_ft)
	model_ft = model_ft.to(device)

	criterion = nn.CrossEntropyLoss()

	# Observe that all parameters are being optimized
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=float(args.lr), lr_momentum=float(args.lr_momentum))

	# Decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma))

	#Get list of dataset transforms
	data_transforms = transforms(params['patch_size'], channel_means, channel_stds, args.colourjitter)

	for partition in training_partitions:
		stats = {'hyperparameters': vars(args), 'trainval_split': trainval_split, 'train': [], 'val': []}
		print("Starting training for partition size: " + str(partition))
		train_cases = cases[-1 - partition:-1]

		train_dataset = torch.utils.data.ConcatDataset([EmptyDirAcceptingImageFolder(os.path.join(
			data_dir, trainCase), data_transforms['train']) for trainCase in train_cases])
		val_dataset = torch.utils.data.ConcatDataset([EmptyDirAcceptingImageFolder(os.path.join(
			data_dir, valCase), data_transforms['val']) for valCase in val_cases])

		image_datasets = {'train': train_dataset, 'val': val_dataset}
		dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

		dataloaders = {}

		if args.useimbalanceddatasetsampler:
			dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=params['batch_size'], sampler=ImbalancedDatasetSampler(image_datasets['train']), num_workers=16)
		else: 
			dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=params['batch_size'], num_workers=16)

		dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=params['batch_size'], num_workers=16)

		since = time.time()

		model_ft, params = train_model(model_ft, params, device, criterion, optimizer_ft, exp_lr_scheduler, class_names, dataloaders, num_epochs=int(args.num_epochs), silent=args.silent)

		time_elapsed = time.time() - since
		print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

		pickle.dump(stats, open(os.path.join(args.output, 'training-' + args.stain + '-' + str(args.network) + '-' + str(len(train_cases)) + '-' + str(len(val_cases)) + '_' + str(now.day) + '-' + str(now.month) + '-' + str(now.year) + '.pickle'), 'wb'))
		torch.save(model_ft, os.path.join(args.output, 'trained_models', 'partSize-' + args.stain + '-' + str(len(train_cases)) + '-' + str(len(val_cases)) + '-' + str(args.network) + '_' + str(now.day) + '-' + str(now.month) + '-' + str(now.year) + '_ft.pt'))
		torch.cuda.empty_cache()