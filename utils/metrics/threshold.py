# Adam Berman and Marcel Gehrung
#
# Calibration script to test fully trained model. Only run on the validation set to find the optimal threshold
#

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def parse_args():
	parser = argparse.ArgumentParser(description='Calibrate on test set.')
	
	parser.add_argument('--stain', type=str, required=True, help='he or p53')
	parser.add_argument('--csv', type=str, required=True, help='Path to CSV file to plot')

	parser.add_argument('--architecture', type=str, help='Model architectures to plot (str)')
	parser.add_argument('--output', type=str, default='figures/results', help='path to save figures to')

	#thresholds to consider
	parser.add_argument('--lower_thresh', type=str, default='0.0', help='Lower thresholds to plot and consider (str separated by comma)')
	parser.add_argument('--upper thresh', type=str, default='1.0', help='Upper thresholds to plot and consider (str separated by comma)')
	parser.add_argument('--step', type=str, default='0.05', help='step values to consider between thresholds (str separated by comma)')
	
	#plots to produce
	parser.add_argument('--pr', action='store_true', help='produce precision-recall plot')
	parser.add_argument('--auc', action='store_true', help='produce area under the curve plot')
	parser.add_argument('--roc', action='store_true', help='produce receiver operating characteristic curve plot')
	parser.add_argument('--auprc', action='store_true', help='produce area under the precision recall curve plot')
	parser.add_argument('--thresh', action='store_true', help='produce area under the precision recall threshold plot')
	
	parser.add_argument('--format', type=str, default='.png', help='formats to output plots to')

	args = parser.parse_args()

	if args.stain not in ['he', 'p53', 'tff3']:
		raise NotImplementedError('-stain argument must be either "he", "p53", "tff3"')

	return args

def generate_stats():
	auc_data = []
	auprc_data = []
	fpr_data = []
	tpr_data = []
	precision_data = []
	recall_data = []

	binary_recall = []
	binary_precision = []
	binary_f1 = []

	for threshold in thresh_data:
		auc_data.append(roc_auc_score(df[calibration_label], df[threshold]))
		auprc_data.append(average_precision_score(df[calibration_label], df[threshold]))
		
		fpr, tpr, thresholds = roc_curve(df[calibration_label], df[threshold])
		precision, recall, thresholds = precision_recall_curve(df[calibration_label], df[threshold])

		fpr_data.append(fpr)
		tpr_data.append(tpr)
		precision_data.append(precision)
		recall_data.append(recall)

		pred = df[threshold].tolist()
		gt = df[calibration_label].tolist()
		binary_pred = np.where(pred > 0, 1, 0)

		binary_precision.append(precision_score(gt, binary_pred))
		binary_recall.append(recall_score(gt, binary_pred))
		binary_f1.append(f1_score(gt, binary_pred))

	return auc_data, auprc_data, fpr_data, tpr_data, precision_data, recall_data, binary_recall, binary_precision, binary_f1

# PLOTS
def precision_recall_plots(df):
	# # Plot precision and recall across thresholds
	fig = plt.figure()
	ax = fig.add_subplot()

	df['Thresh'] = df['Thresh'].astype(float)
	df.plot(x='Thresh', y=['Precision', 'Recall'], ax=ax)

	ax.set(xlabel='Threshold', ylabel='Value')
	ax.set_title('Precision and recall across thresholds')
	# ax.set_xticks(rotation=90)
	ax.legend(loc = 'lower right')
	ax.set_xlim([0.97, 1])
	ax.set_ylim([0, 1])
	ax.spines[['top', 'right']].set_visible(False)
	return fig

def auc_thresh_plot(auc_probs, thresh_prob, biomarker):
	# Plot AUC at each threshold to determine best probability threshold
	fig = plt.figure(figsize=(7.5,6))
	ax = fig.add_subplot()
	ax.plot(thresh_prob, auc_probs['prob'])
	# ax.legend(loc='lower center', prop={'size': 12})
	ax.set_xlim(-0.05, 1.05)
	# ax.set_ylim(0.6, 1.0)
	ax.set_xlabel('Probability threshold for determination\nof number of tiles with ' + biomarker)
	ax.set_ylabel('AUC-ROC for Cytosponge ' + biomarker + ' detection with\nthresholded number of tiles')
	ax.spines[['top', 'right']].set_visible(False)
	return fig

def roc_thresh_plot(cutoffs, auc_probs, auc_plotting, biomarker):
	# Plot ROC curve for best AUC probability threshold
	fig = plt.figure()
	ax = fig.add_subplot()
	ax.set_title('ROC with ' + biomarker + ' probability threshold of ' + str(cutoffs['tile_thresh']))
	ax.plot(auc_plotting['fpr'], auc_plotting['tpr'], 'b', label = 'AUC = %0.2f' % max(auc_probs['prob']))
	ax.legend(loc = 'lower right')
	ax.plot([0, 1], [0, 1],'r--')
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])
	ax.set_ylabel('True Positive Rate')
	ax.set_xlabel('False Positive Rate')
	ax.spines[['top', 'right']].set_visible(False)
	return fig

def auprc_curve_plot(auprc_cutoffs, auprc_probs, auprc_plotting, biomarker):
	#Plot AUPRC curve for best AUC probability threshold
	fig = plt.figure()
	ax = fig.add_subplot()
	ax.set_title('AUPRC with ' + biomarker + ' probability threshold of ' + str(auprc_cutoffs['tile_thresh']))
	ax.plot(auprc_plotting['recall'], auprc_plotting['precision'], 'b', label = 'AUC = %0.3f' % max(auprc_probs['prob']))
	ax.legend(loc = 'lower right')
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])
	ax.set_xlabel('Recall')
	ax.set_ylabel('Precision')
	ax.spines[['top', 'right']].set_visible(False)
	return fig

def auprc_thresh_plot(auprc_probs, thresh_prob, biomarker):
	# Plot AUPRC at each threshold to determine best probability threshold
	fig = plt.figure(figsize=(7.5,6))
	ax = fig.add_subplot()
	plt.plot(thresh_prob, auprc_probs['prob'])
	# ax.legend(loc='lower center', prop={'size': 12})
	ax.set_xlim(-0.05, 1.05)
	# ax.set_ylim(0.6, 1.0)
	ax.set_xlabel('Probability threshold for determination\nof number of tiles with ' + biomarker)
	ax.set_ylabel('AUPRC for Cytosponge ' + biomarker + ' detection with\nthresholded number of tiles')
	ax.spines[['top', 'right']].set_visible(False)
	return fig

if __name__ == '__main__':
	args = parse_args()

	plt.style.use('figures/journal-style.mplstyle')

	if args.stain == 'he':
		biomarker = 'atypia'
		cell = 'Gastric '
		calibration_label = 'Pathologist ' + biomarker
	elif args.stain == 'p53':
		biomarker = 'p53'
		cell = 'IM postive '
		calibration_label = 'Pathologist ' + biomarker
	else:
		biomarker = 'tff3'
		cell = 'TFF3 positive '
		calibration_label = 'Endoscopy (at least C1 or M3) + Biopsy (IM)'

	csv = args.csv
	df = pd.read_csv(csv)

	if not os.path.exists(args.output):
		os.makedirs(args.output)
	output_folder = os.path.join(args.output, args.architecture)
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	thresholdList = []
	cutoff_prob = []
	auprc_cutoff_prob = []

	lower_thresholds = [float(lt) for lt in list(args.lower_thresh.split(','))]
	upper_thresholds = [float(ut) for ut in list(args.upper_thresh.split(','))]
	steps = [float(st) for st in list(args.step.split(','))]
	if not len(upper_thresholds) == len(lower_thresholds) == len(steps):
		raise Warning('Thresholds and steps should all be the same length')

	thresh_prob = np.empty(shape=0)
	for lt, ut, st in zip(lower_thresholds, upper_thresholds, steps):
		thresh_prob = np.append(thresh_prob, np.arange(lt, ut, st))
	thresh_prob[0] = min(lower_thresholds)
	thresh_prob[-1] = max(upper_thresholds)

	thresh_data = df.loc[:, cell + 'count (> ' + str(thresh_prob) + ')':cell + 'count (> ' + str(thresh_prob) + ')']

	cutoffs = {}
	auc_probs = {}
	auc_plotting = {}
	auprc_cutoffs = {}
	auprc_probs = {}
	auprc_plotting = {}

	auc_data = []
	auprc_data = []
	fpr_data = []
	tpr_data = []
	precision_data = []
	recall_data = []

	binary_recall = []
	binary_precision = []
	binary_f1 = []

	for threshold in thresh_data:
		auc_data.append(roc_auc_score(df[calibration_label], df[threshold]))
		auprc_data.append(average_precision_score(df[calibration_label], df[threshold]))
		
		fpr, tpr, thresholds = roc_curve(df[calibration_label], df[threshold])
		precision, recall, thresholds = precision_recall_curve(df[calibration_label], df[threshold])

		fpr_data.append(fpr)
		tpr_data.append(tpr)
		precision_data.append(precision)
		recall_data.append(recall)

		pred = (df[threshold] > 0).tolist()
		gt = df[calibration_label].tolist()

		binary_precision.append(precision_score(gt, pred))
		binary_recall.append(recall_score(gt, pred))
		binary_f1.append(f1_score(gt, pred))

	pd.set_option('display.max_rows', None)
	thresh_prec_rec_df = pd.DataFrame(list(zip([str(p) for p in thresh_prob], binary_precision, binary_recall, binary_f1)), columns=['Thresh', 'Precision', 'Recall', 'F1'])
	thresh_prec_rec_melted_df = thresh_prec_rec_df.melt(id_vars=['Thresh'], value_vars=['Precision', 'Recall'])

	max_auc = max(auc_data)
	max_auc_idx = auc_data.index(max_auc)
	max_auc_data = [i for i, j in enumerate(auc_data) if j == max_auc]
	print('Probability: ' + str(thresh_prob[max_auc_data[0]]), 'AUC: ' + str(auc_data[max_auc_data[0]]))

	cutoff_prob.append(round(thresh_prob[max_auc_data[0]],6))
	cutoffs['tile_thresh'] = round(thresh_prob[max_auc_data[0]], 7)
	auc_probs['prob'] = auc_data

	auc_plotting['fpr'] = fpr[max_auc_idx]
	auc_plotting['tpr'] = tpr_data[max_auc_idx]

	print('-'*10)

	max_auprc = max(auprc_data)
	max_auprc_idx = auprc_data.index(max_auprc)
	max_auprc_data = [i for i, j in enumerate(auprc_data) if j == max_auprc]
	print('Probability: ' + str(thresh_prob[max_auprc_data[0]]), 'AUPRC: ' + str(auprc_data[max_auprc_data[0]]))
	auprc_cutoff_prob.append(round(thresh_prob[max_auprc_data[0]],6))
	auprc_cutoffs['tile_thresh'] = thresh_prob[max_auprc_data[0]]
	auprc_probs['prob'] = auprc_data

	auprc_plotting['precision'] = precision_data[max_auprc_idx]
	auprc_plotting['recall'] = recall_data[max_auprc_idx]

	if args.pr:
		pr_fig = precision_recall_plots(thresh_prec_rec_melted_df)
		pr_fig.savefig(os.path.join(output_folder, 'pr_curve' + biomarker.upper() + args.format))
	if args.auc:
		auc_fig = auc_thresh_plot(auc_probs, thresh_prob, biomarker)
		auc_fig.savefig(os.path.join(output_folder, 'auc_prob_threshold' + biomarker.upper() + args.format)) #57casetrainval
	if args.roc:
		roc_fig = roc_thresh_plot(cutoffs, auc_probs, auc_plotting, biomarker)
		roc_fig.savefig(os.path.join(output_folder, 'roc_curve' + biomarker.upper() + args.format))
	if args.auprc:
		auprc_curve_fig = auprc_curve_plot(auprc_cutoffs, auprc_probs, auprc_plotting, biomarker)
		auprc_curve_fig.savefig(os.path.join(output_folder, 'auprc_curve' + biomarker.upper() + args.format))
	if args.thresh:
		auprc_thresh_fig = auprc_thresh_plot(auprc_probs, thresh_prob, biomarker)
		auprc_thresh_fig.savefig(os.path.join(output_folder, 'auprc_prob_threshold' + biomarker.upper() + args.format))

	print(cutoffs)

	with open(os.path.join(os.path.join(os.path.split(args.csv)[0]), args.stain+'_thresholds.pickle'), 'wb') as f:
		pickle.dump(cutoffs, f)