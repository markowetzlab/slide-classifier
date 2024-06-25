import pandas as pd
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
				classification_report, confusion_matrix)
	if args.stats:
		df = pd.DataFrame.from_dict(data_list)

		gt_col = df['Ground Truth Label']
		gt = gt_col.tolist()

		pred_col = df[ranked_class]
		print(f'\nLCP Tile Threshold ({lcp_threshold})')
		df['LCP'] = pred_col.gt(int(lcp_threshold)).astype(int)
		pred = df[f'LCP'].tolist()

		if args.process_list is not None:
			tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
			auc = roc_auc_score(gt, pred)
			precision = precision_score(gt, pred)
			recall = recall_score(gt, pred)
			f1 = f1_score(gt, pred)
			
			print('AUC: ', auc)
			print('Sensitivity: ', recall)
			print('Specificity: ', tn/(tn+fp))
			print('Precision: ', precision)
			print('F1: ', f1, '\n')

			print(f'CM \tGT Positive\tGT Negative')
			print(f'Pred Positive\t{tp}\t{fp}')
			print(f'Pred Negative\t{fn}\t{tn}')
			print(classification_report(gt, pred))
		else:
			print(df['LCP'].value_counts())

		print(f'\nHCP Tile Threshold ({hcp_threshold})')
		df['HCP'] = pred_col.gt(int(hcp_threshold)).astype(int)
		pred = df[f'HCP'].tolist()

		if args.process_list is not None:
			tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
			auc = roc_auc_score(gt, pred)
			precision = precision_score(gt, pred)
			recall = recall_score(gt, pred)
			f1 = f1_score(gt, pred)
			
			print('AUC: ', auc)
			print('Sensitivity: ', recall)
			print('Specificity: ', tn/(tn+fp))
			print('Precision: ', precision)
			print('F1: ', f1, '\n')

			print(f'CM \tGT Positive\tGT Negative')
			print(f'Pred Positive\t{tp}\t{fp}')
			print(f'Pred Negative\t{fn}\t{tn}')

			print(classification_report(gt, pred))
		else:
			print(df['HCP'].value_counts())

		results = [
			(df['LCP'] == 0) & (df['HCP'] == 0),
			(df['LCP'] == 1) & (df['HCP'] == 0),
			(df['LCP'] == 1) & (df['HCP'] == 1)
		]
		values = ['High Confidence Negative', 'Low Confidence Positive', 'High Confidence Positive']
		df['Result'] = np.select(results, values)
		print('Predictions: \n', df['Result'].value_counts())

		if args.csv:
			df.to_csv(os.path.join(directories['save_dir'], 'results.csv'), index=False)