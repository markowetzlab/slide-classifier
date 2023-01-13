# Adam Berman

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

atypia_val_results = pd.read_csv('/home/cri.camres.org/berman01/progression-detection/data/FINAL_RESULTS/ATYPIA_VAL_0-99993.csv')
atypia_test_results = pd.read_csv('/home/cri.camres.org/berman01/progression-detection/data/FINAL_RESULTS/ATYPIA_TEST_0-99993.csv')

p53_val_results = pd.read_csv('/home/cri.camres.org/berman01/progression-detection/data/FINAL_RESULTS/P53_VAL_0-999.csv')
p53_test_results = pd.read_csv('/home/cri.camres.org/berman01/progression-detection/data/FINAL_RESULTS/P53_TEST_0-999.csv')

#print(p53_val_results)
#print(p53_test_results)

# -------------- VAL ------------------

atypia_val_results['endo_dysplasia'] = np.where(((atypia_val_results['endoscopy'] == 'HGD/IMC') | (atypia_val_results['endoscopy'] == 'LGD')), 1, 0)
atypia_val_results['endo_dysplasia_including_ind'] = np.where(((atypia_val_results['endoscopy'] == 'HGD/IMC') | (atypia_val_results['endoscopy'] == 'LGD') | (atypia_val_results['endoscopy'] == 'IND')), 1, 0)
atypia_val_results['atypia_tile_count_greater_than_0'] = np.where(atypia_val_results['atypia_tile_count'] > 0, 1, 0)

atypia_tile_counts = atypia_val_results['atypia_tile_count'].tolist()
atypia_tile_count_greater_than_0 = atypia_val_results['atypia_tile_count_greater_than_0'].tolist()
atypia_ground_truth_after_ai_review = atypia_val_results['atypia_ground_truth_after_ai_review'].tolist()
atypia_triage_scheme = []
atypia_hcn_triage_threshold = 0
atypia_hcp_triage_threshold = 10
for i, atypia_tile_count in enumerate(atypia_tile_counts):
    if atypia_tile_count <= atypia_hcn_triage_threshold:
        atypia_triage_scheme.append(atypia_tile_count_greater_than_0[i])
    elif atypia_tile_count < atypia_hcp_triage_threshold:
        atypia_triage_scheme.append(atypia_ground_truth_after_ai_review[i])
    else:
        atypia_triage_scheme.append(atypia_tile_count_greater_than_0[i])
atypia_val_results['atypia_triage_scheme'] = atypia_triage_scheme
#print(atypia_triage_scheme)
#print(len(atypia_triage_scheme))
#quit()


p53_val_results['endo_dysplasia'] = np.where(((p53_val_results['endoscopy'] == 'HGD/IMC') | (p53_val_results['endoscopy'] == 'LGD')), 1, 0)
p53_val_results['endo_dysplasia_including_ind'] = np.where(((p53_val_results['endoscopy'] == 'HGD/IMC') | (p53_val_results['endoscopy'] == 'LGD') | (p53_val_results['endoscopy'] == 'IND')), 1, 0)
p53_val_results['p53_tile_count_greater_than_0'] = np.where(p53_val_results['p53_tile_count'] > 0, 1, 0)

p53_tile_counts = p53_val_results['p53_tile_count'].tolist()
p53_tile_count_greater_than_0 = p53_val_results['p53_tile_count_greater_than_0'].tolist()
p53_ground_truth_after_ai_review = p53_val_results['p53_ground_truth_after_ai_review'].tolist()
p53_triage_scheme = []
p53_hcn_triage_threshold = 0
p53_hcp_triage_threshold = 2
for i, p53_tile_count in enumerate(p53_tile_counts):
    if p53_tile_count <= p53_hcn_triage_threshold:
        p53_triage_scheme.append(p53_tile_count_greater_than_0[i])
    elif p53_tile_count < p53_hcp_triage_threshold:
        p53_triage_scheme.append(p53_ground_truth_after_ai_review[i])
    else:
        p53_triage_scheme.append(p53_tile_count_greater_than_0[i])
p53_val_results['p53_triage_scheme'] = p53_triage_scheme

# Only positive subset dataframe

p53_val_results_positive_subset = p53_val_results[p53_val_results['p53_tile_count_greater_than_0'] + p53_val_results['p53_ground_truth_after_ai_review'] >= 1]
#print(p53_val_results_positive_subset)
#quit()


# Combined dataframe

atypiap53_val_results = atypia_val_results.merge(p53_val_results, on='case')
#print("COMBINED VAL ROWS:", len(atypiap53_val_results.index))
atypiap53_val_results['atypiap53_ground_truth_no_ai_review'] = np.where(((atypiap53_val_results['atypia_ground_truth_no_ai_review'] == 1) | (atypiap53_val_results['p53_ground_truth_no_ai_review'] == 1)), 1, 0)
atypiap53_val_results['atypiap53_ground_truth_after_ai_review'] = np.where(((atypiap53_val_results['atypia_ground_truth_after_ai_review'] == 1) | (atypiap53_val_results['p53_ground_truth_after_ai_review'] == 1)), 1, 0)
atypiap53_val_results['atypiap53_tile_count_greater_than_0'] = np.where(((atypiap53_val_results['atypia_tile_count_greater_than_0'] == 1) | (atypiap53_val_results['p53_tile_count_greater_than_0'] == 1)), 1, 0)
atypiap53_val_results['atypiap53_triage_scheme'] = np.where(((atypiap53_val_results['atypia_triage_scheme'] == 1) | (atypiap53_val_results['p53_triage_scheme'] == 1)), 1, 0)
#print(atypiap53_val_results)
#quit()
triage_class_letter = []
for i in range(len(atypiap53_val_results.index)):
    if atypiap53_val_results['atypia_tile_count'][i] >= atypia_hcp_triage_threshold and atypiap53_val_results['p53_tile_count'][i] >= p53_hcp_triage_threshold:
        triage_class_letter.append('A')
    elif atypiap53_val_results['atypia_tile_count'][i] >= atypia_hcp_triage_threshold and (atypiap53_val_results['p53_tile_count'][i] > p53_hcn_triage_threshold and atypiap53_val_results['p53_tile_count'][i] < p53_hcp_triage_threshold):
        triage_class_letter.append('B')
    elif atypiap53_val_results['atypia_tile_count'][i] >= atypia_hcp_triage_threshold and atypiap53_val_results['p53_tile_count'][i] <= p53_hcn_triage_threshold:
        triage_class_letter.append('C')
    elif (atypiap53_val_results['atypia_tile_count'][i] > atypia_hcn_triage_threshold and atypiap53_val_results['atypia_tile_count'][i] < atypia_hcp_triage_threshold) and atypiap53_val_results['p53_tile_count'][i] >= p53_hcp_triage_threshold:
        triage_class_letter.append('D')
    elif (atypiap53_val_results['atypia_tile_count'][i] > atypia_hcn_triage_threshold and atypiap53_val_results['atypia_tile_count'][i] < atypia_hcp_triage_threshold) and (atypiap53_val_results['p53_tile_count'][i] > p53_hcn_triage_threshold and atypiap53_val_results['p53_tile_count'][i] < p53_hcp_triage_threshold):
        triage_class_letter.append('E')
    elif (atypiap53_val_results['atypia_tile_count'][i] > atypia_hcn_triage_threshold and atypiap53_val_results['atypia_tile_count'][i] < atypia_hcp_triage_threshold) and atypiap53_val_results['p53_tile_count'][i] <= p53_hcn_triage_threshold:
        triage_class_letter.append('F')
    elif atypiap53_val_results['atypia_tile_count'][i] <= atypia_hcn_triage_threshold and atypiap53_val_results['p53_tile_count'][i] >= p53_hcp_triage_threshold:
        triage_class_letter.append('G')
    elif atypiap53_val_results['atypia_tile_count'][i] <= atypia_hcn_triage_threshold and (atypiap53_val_results['p53_tile_count'][i] > p53_hcn_triage_threshold and atypiap53_val_results['p53_tile_count'][i] < p53_hcp_triage_threshold):
        triage_class_letter.append('H')
    elif atypiap53_val_results['atypia_tile_count'][i] <= atypia_hcn_triage_threshold and atypiap53_val_results['p53_tile_count'][i] <= p53_hcn_triage_threshold:
        triage_class_letter.append('I')
    else:
        raise Warning('Row does not fit into a triage class letter.')
atypiap53_val_results['atypiap53_triage_class_letter'] = triage_class_letter




# ------------------- TEST -----------------------


atypia_test_results['endo_dysplasia'] = np.where(((atypia_test_results['endoscopy'] == 'HGD/IMC') | (atypia_test_results['endoscopy'] == 'LGD')), 1, 0)
atypia_test_results['endo_dysplasia_including_ind'] = np.where(((atypia_test_results['endoscopy'] == 'HGD/IMC') | (atypia_test_results['endoscopy'] == 'LGD') | (atypia_test_results['endoscopy'] == 'IND')), 1, 0)
atypia_test_results['atypia_tile_count_greater_than_0'] = np.where(atypia_test_results['atypia_tile_count'] > 0, 1, 0)

atypia_tile_counts = atypia_test_results['atypia_tile_count'].tolist()
atypia_tile_count_greater_than_0 = atypia_test_results['atypia_tile_count_greater_than_0'].tolist()
atypia_ground_truth_after_ai_review = atypia_test_results['atypia_ground_truth_after_ai_review'].tolist()
atypia_triage_scheme = []
atypia_hcn_triage_threshold = 0
atypia_hcp_triage_threshold = 10
for i, atypia_tile_count in enumerate(atypia_tile_counts):
    if atypia_tile_count <= atypia_hcn_triage_threshold:
        atypia_triage_scheme.append(atypia_tile_count_greater_than_0[i])
    elif atypia_tile_count < atypia_hcp_triage_threshold:
        atypia_triage_scheme.append(atypia_ground_truth_after_ai_review[i])
    else:
        atypia_triage_scheme.append(atypia_tile_count_greater_than_0[i])
atypia_test_results['atypia_triage_scheme'] = atypia_triage_scheme
#print(atypia_triage_scheme)
#print(len(atypia_triage_scheme))
#quit()


p53_test_results['endo_dysplasia'] = np.where(((p53_test_results['endoscopy'] == 'HGD/IMC') | (p53_test_results['endoscopy'] == 'LGD')), 1, 0)
p53_test_results['endo_dysplasia_including_ind'] = np.where(((p53_test_results['endoscopy'] == 'HGD/IMC') | (p53_test_results['endoscopy'] == 'LGD') | (p53_test_results['endoscopy'] == 'IND')), 1, 0)
p53_test_results['p53_tile_count_greater_than_0'] = np.where(p53_test_results['p53_tile_count'] > 0, 1, 0)

p53_tile_counts = p53_test_results['p53_tile_count'].tolist()
p53_tile_count_greater_than_0 = p53_test_results['p53_tile_count_greater_than_0'].tolist()
p53_ground_truth_after_ai_review = p53_test_results['p53_ground_truth_after_ai_review'].tolist()
p53_triage_scheme = []
p53_hcn_triage_threshold = 0
p53_hcp_triage_threshold = 2
for i, p53_tile_count in enumerate(p53_tile_counts):
    if p53_tile_count <= p53_hcn_triage_threshold:
        p53_triage_scheme.append(p53_tile_count_greater_than_0[i])
    elif p53_tile_count < p53_hcp_triage_threshold:
        p53_triage_scheme.append(p53_ground_truth_after_ai_review[i])
    else:
        p53_triage_scheme.append(p53_tile_count_greater_than_0[i])
p53_test_results['p53_triage_scheme'] = p53_triage_scheme

# Only positive subset dataframe

p53_test_results_positive_subset = p53_test_results[p53_test_results['p53_tile_count_greater_than_0'] + p53_test_results['p53_ground_truth_after_ai_review'] >= 1]
#print(p53_test_results_positive_subset)
#quit()

# Combined dataframe

atypiap53_test_results = atypia_test_results.merge(p53_test_results, on='case')
#print(atypiap53_test_results.columns)
#quit()
atypiap53_test_results['atypiap53_ground_truth_no_ai_review'] = np.where(((atypiap53_test_results['atypia_ground_truth_no_ai_review'] == 1) | (atypiap53_test_results['p53_ground_truth_no_ai_review'] == 1)), 1, 0)
atypiap53_test_results['atypiap53_ground_truth_after_ai_review'] = np.where(((atypiap53_test_results['atypia_ground_truth_after_ai_review'] == 1) | (atypiap53_test_results['p53_ground_truth_after_ai_review'] == 1)), 1, 0)
atypiap53_test_results['atypiap53_tile_count_greater_than_0'] = np.where(((atypiap53_test_results['atypia_tile_count_greater_than_0'] == 1) | (atypiap53_test_results['p53_tile_count_greater_than_0'] == 1)), 1, 0)
atypiap53_test_results['atypiap53_triage_scheme'] = np.where(((atypiap53_test_results['atypia_triage_scheme'] == 1) | (atypiap53_test_results['p53_triage_scheme'] == 1)), 1, 0)

#print(atypiap53_test_results['atypia_tile_count'])
#for i in range(len(atypiap53_test_results.index)):
#    print(atypiap53_test_results['atypia_tile_count'][i])

triage_class_letter = []
for i in range(len(atypiap53_test_results.index)):
    if atypiap53_test_results['atypia_tile_count'][i] >= atypia_hcp_triage_threshold and atypiap53_test_results['p53_tile_count'][i] >= p53_hcp_triage_threshold:
        triage_class_letter.append('A')
    elif atypiap53_test_results['atypia_tile_count'][i] >= atypia_hcp_triage_threshold and (atypiap53_test_results['p53_tile_count'][i] > p53_hcn_triage_threshold and atypiap53_test_results['p53_tile_count'][i] < p53_hcp_triage_threshold):
        triage_class_letter.append('B')
    elif atypiap53_test_results['atypia_tile_count'][i] >= atypia_hcp_triage_threshold and atypiap53_test_results['p53_tile_count'][i] <= p53_hcn_triage_threshold:
        triage_class_letter.append('C')
    elif (atypiap53_test_results['atypia_tile_count'][i] > atypia_hcn_triage_threshold and atypiap53_test_results['atypia_tile_count'][i] < atypia_hcp_triage_threshold) and atypiap53_test_results['p53_tile_count'][i] >= p53_hcp_triage_threshold:
        triage_class_letter.append('D')
    elif (atypiap53_test_results['atypia_tile_count'][i] > atypia_hcn_triage_threshold and atypiap53_test_results['atypia_tile_count'][i] < atypia_hcp_triage_threshold) and (atypiap53_test_results['p53_tile_count'][i] > p53_hcn_triage_threshold and atypiap53_test_results['p53_tile_count'][i] < p53_hcp_triage_threshold):
        triage_class_letter.append('E')
    elif (atypiap53_test_results['atypia_tile_count'][i] > atypia_hcn_triage_threshold and atypiap53_test_results['atypia_tile_count'][i] < atypia_hcp_triage_threshold) and atypiap53_test_results['p53_tile_count'][i] <= p53_hcn_triage_threshold:
        triage_class_letter.append('F')
    elif atypiap53_test_results['atypia_tile_count'][i] <= atypia_hcn_triage_threshold and atypiap53_test_results['p53_tile_count'][i] >= p53_hcp_triage_threshold:
        triage_class_letter.append('G')
    elif atypiap53_test_results['atypia_tile_count'][i] <= atypia_hcn_triage_threshold and (atypiap53_test_results['p53_tile_count'][i] > p53_hcn_triage_threshold and atypiap53_test_results['p53_tile_count'][i] < p53_hcp_triage_threshold):
        triage_class_letter.append('H')
    elif atypiap53_test_results['atypia_tile_count'][i] <= atypia_hcn_triage_threshold and atypiap53_test_results['p53_tile_count'][i] <= p53_hcn_triage_threshold:
        triage_class_letter.append('I')
    else:
        raise Warning('Row does not fit into a triage class letter.')
atypiap53_test_results['atypiap53_triage_class_letter'] = triage_class_letter

#p53_val_results['endo_dysplasia'] = np.where(((p53_val_results['endoscopy'] == 'HGD/IMC') | (p53_val_results['endoscopy'] == 'LGD')), 1, 0)
#p53_val_results['endo_dysplasia_including_ind'] = np.where(((p53_val_results['endoscopy'] == 'HGD/IMC') | (p53_val_results['endoscopy'] == 'LGD') | (p53_val_results['endoscopy'] == 'IND')), 1, 0)
#p53_val_results['p53_tile_count_greater_than_0'] = np.where(p53_val_results['p53_tile_count'] > 1, 1, 0)

#p53_test_results['endo_dysplasia'] = np.where(((p53_test_results['endoscopy'] == 'HGD/IMC') | (p53_test_results['endoscopy'] == 'LGD')), 1, 0)
#p53_test_results['endo_dysplasia_including_ind'] = np.where(((p53_test_results['endoscopy'] == 'HGD/IMC') | (p53_test_results['endoscopy'] == 'LGD') | (p53_test_results['endoscopy'] == 'IND')), 1, 0)
#p53_test_results['p53_tile_count_greater_than_0'] = np.where(p53_test_results['p53_tile_count'] > 1, 1, 0)

#print(p53_test_results)

# VALIDATION ANALYSES

# Evaluate model's performance relative to pathologist ground truth

# Evaluate model's performance relative to endoscopy ground truth


# TEST ANALYSES

# Evaluate model's performance relative to pathologist ground truth
print()
print('----------------------ATYPIA VAL----------------------')
print()
print('Atypia val model vs. pre-AI pathologist ground truth')
print(classification_report(atypia_val_results['atypia_ground_truth_no_ai_review'], atypia_val_results['atypia_tile_count_greater_than_0']))
print()
print('Atypia val model vs. post-AI pathologist ground truth')
print(classification_report(atypia_val_results['atypia_ground_truth_after_ai_review'], atypia_val_results['atypia_tile_count_greater_than_0']))
print()
print('Atypia val model vs. endoscopy ground truth')
print(classification_report(atypia_val_results['endo_dysplasia'], atypia_val_results['atypia_tile_count_greater_than_0']))
print()
print('Atypia val pre-AI pathologist vs. endoscopy ground truth')
print(classification_report(atypia_val_results['endo_dysplasia'], atypia_val_results['atypia_ground_truth_no_ai_review']))
print()
print('Atypia val post-AI pathologist vs. endoscopy ground truth')
print(classification_report(atypia_val_results['endo_dysplasia'], atypia_val_results['atypia_ground_truth_after_ai_review']))
print()
print('Atypia val atypia-only triage system vs. post-AI pathologist ground truth')
print(classification_report(atypia_val_results['atypia_ground_truth_after_ai_review'], atypia_val_results['atypia_triage_scheme']))
print()
print('Atypia val atypia-only triage system vs. endoscopy ground truth')
print(classification_report(atypia_val_results['endo_dysplasia'], atypia_val_results['atypia_triage_scheme']))

print()
print('----------------------P53 VAL----------------------')
print()
print('P53 val model vs. pre-AI pathologist ground truth')
print(classification_report(p53_val_results['p53_ground_truth_no_ai_review'], p53_val_results['p53_tile_count_greater_than_0']))
print()
print('P53 val model vs. post-AI pathologist ground truth')
print(classification_report(p53_val_results['p53_ground_truth_after_ai_review'], p53_val_results['p53_tile_count_greater_than_0']))
print()
print('P53 val model vs. endoscopy ground truth')
print(classification_report(p53_val_results['endo_dysplasia'], p53_val_results['p53_tile_count_greater_than_0']))
print()
print('P53 val pre-AI pathologist vs. endoscopy ground truth')
print(classification_report(p53_val_results['endo_dysplasia'], p53_val_results['p53_ground_truth_no_ai_review']))
print()
print('P53 val post-AI pathologist vs. endoscopy ground truth')
print(classification_report(p53_val_results['endo_dysplasia'], p53_val_results['p53_ground_truth_after_ai_review']))
print()
print('P53 val P53-only triage system vs. post-AI pathologist ground truth')
print(classification_report(p53_val_results['p53_ground_truth_after_ai_review'], p53_val_results['p53_triage_scheme']))
print()
print('P53 val P53-only triage system vs. endoscopy ground truth')
print(classification_report(p53_val_results['endo_dysplasia'], p53_val_results['p53_triage_scheme']))
print()
print('P53 val P53-only triage system vs. endoscopy ground truth [POSITIVE SUBSET]')
print(classification_report(p53_val_results_positive_subset['endo_dysplasia'], p53_val_results_positive_subset['p53_triage_scheme']))


print()
print('----------------------COMBINED VAL----------------------')
print()
print('Atypia-P53 val model vs. pre-AI pathologist ground truth')
print(classification_report(atypiap53_val_results['atypiap53_ground_truth_no_ai_review'], atypiap53_val_results['atypiap53_tile_count_greater_than_0']))
print()
print('Atypia-P53 val model vs. post-AI pathologist ground truth')
print(classification_report(atypiap53_val_results['atypiap53_ground_truth_after_ai_review'], atypiap53_val_results['atypiap53_tile_count_greater_than_0']))
print()
print('Atypia-P53 val model vs. endoscopy ground truth')
print(classification_report(atypiap53_val_results['endo_dysplasia_x'], atypiap53_val_results['atypiap53_tile_count_greater_than_0']))
print()
print('Atypia-P53 val pre-AI pathologist vs. endoscopy ground truth')
print(classification_report(atypiap53_val_results['endo_dysplasia_x'], atypiap53_val_results['atypiap53_ground_truth_no_ai_review']))
print()
print('Atypia-P53 val post-AI pathologist vs. endoscopy ground truth')
print(classification_report(atypiap53_val_results['endo_dysplasia_x'], atypiap53_val_results['atypiap53_ground_truth_after_ai_review']))
print()
print('Atypia-P53 val Atypia-P53 triage system vs. post-AI pathologist ground truth')
print(classification_report(atypiap53_val_results['atypiap53_ground_truth_after_ai_review'], atypiap53_val_results['atypiap53_triage_scheme']))
print()
print('Atypia-P53 val Atypia-P53 triage system vs. endoscopy ground truth')
print(classification_report(atypiap53_val_results['endo_dysplasia_x'], atypiap53_val_results['atypiap53_triage_scheme']))
#quit()

#print(classification_report(atypia_test_results['p53_ground_truth_after_ai'], atypia_test_results['p53_tile_count_greater_than_0']))

# Evaluate model's performance relative to pathologist ground truth
#print(classification_report(p53_test_results['p53_ground_truth_before_ai'], p53_test_results['p53_tile_count_greater_than_0']))
#print(classification_report(p53_test_results['p53_ground_truth_after_ai'], p53_test_results['p53_tile_count_greater_than_0']))

# Evaluate model's performance relative to endoscopy ground truth

# Compare model's performance on endoscopy ground truth with pre-rereview pathologist's performance on endoscopy ground truth


print()
print('----------------------ATYPIA TEST----------------------')
print()
print('Atypia test model vs. pre-AI pathologist ground truth')
print(classification_report(atypia_test_results['atypia_ground_truth_no_ai_review'], atypia_test_results['atypia_tile_count_greater_than_0']))
print()
print('Atypia test model vs. post-AI pathologist ground truth')
print(classification_report(atypia_test_results['atypia_ground_truth_after_ai_review'], atypia_test_results['atypia_tile_count_greater_than_0']))
print()
print('Atypia test model vs. endoscopy ground truth')
print(classification_report(atypia_test_results['endo_dysplasia'], atypia_test_results['atypia_tile_count_greater_than_0']))
print()
print('Atypia test pre-AI pathologist vs. endoscopy ground truth')
print(classification_report(atypia_test_results['endo_dysplasia'], atypia_test_results['atypia_ground_truth_no_ai_review']))
print()
print('Atypia test post-AI pathologist vs. endoscopy ground truth')
print(classification_report(atypia_test_results['endo_dysplasia'], atypia_test_results['atypia_ground_truth_after_ai_review']))
print()
print('Atypia test atypia-only triage system vs. post-AI pathologist ground truth')
print(classification_report(atypia_test_results['atypia_ground_truth_after_ai_review'], atypia_test_results['atypia_triage_scheme']))
print()
print('Atypia test atypia-only triage system vs. endoscopy ground truth')
print(classification_report(atypia_test_results['endo_dysplasia'], atypia_test_results['atypia_triage_scheme']))

print()
print('----------------------P53 TEST----------------------')
print()
print('P53 test model vs. pre-AI pathologist ground truth')
print(classification_report(p53_test_results['p53_ground_truth_no_ai_review'], p53_test_results['p53_tile_count_greater_than_0']))
print()
print('P53 test model vs. post-AI pathologist ground truth')
print(classification_report(p53_test_results['p53_ground_truth_after_ai_review'], p53_test_results['p53_tile_count_greater_than_0']))
print()
print('P53 test model vs. endoscopy ground truth')
print(classification_report(p53_test_results['endo_dysplasia'], p53_test_results['p53_tile_count_greater_than_0']))
print()
print('P53 test pre-AI pathologist vs. endoscopy ground truth')
print(classification_report(p53_test_results['endo_dysplasia'], p53_test_results['p53_ground_truth_no_ai_review']))
print()
print('P53 test post-AI pathologist vs. endoscopy ground truth')
print(classification_report(p53_test_results['endo_dysplasia'], p53_test_results['p53_ground_truth_after_ai_review']))
print()
print('P53 test P53-only triage system vs. post-AI pathologist ground truth')
print(classification_report(p53_test_results['p53_ground_truth_after_ai_review'], p53_test_results['p53_triage_scheme']))
print()
print('P53 test P53-only triage system vs. endoscopy ground truth')
print(classification_report(p53_test_results['endo_dysplasia'], p53_test_results['p53_triage_scheme']))
print()
print('P53 test P53-only triage system vs. endoscopy ground truth [POSITIVE SUBSET]')
print(classification_report(p53_test_results_positive_subset['endo_dysplasia'], p53_test_results_positive_subset['p53_triage_scheme']))

print()
print('----------------------COMBINED TEST----------------------')
print()
print('Atypia-P53 test model vs. pre-AI pathologist ground truth')
print(classification_report(atypiap53_test_results['atypiap53_ground_truth_no_ai_review'], atypiap53_test_results['atypiap53_tile_count_greater_than_0']))
print()
print('Atypia-P53 test model vs. post-AI pathologist ground truth')
print(classification_report(atypiap53_test_results['atypiap53_ground_truth_after_ai_review'], atypiap53_test_results['atypiap53_tile_count_greater_than_0']))
print()
print('Atypia-P53 test model vs. endoscopy ground truth')
print(classification_report(atypiap53_test_results['endo_dysplasia_x'], atypiap53_test_results['atypiap53_tile_count_greater_than_0']))
print()
print('Atypia-P53 test pre-AI pathologist vs. endoscopy ground truth')
print(classification_report(atypiap53_test_results['endo_dysplasia_x'], atypiap53_test_results['atypiap53_ground_truth_no_ai_review']))
print()
print('Atypia-P53 test post-AI pathologist vs. endoscopy ground truth')
print(classification_report(atypiap53_test_results['endo_dysplasia_x'], atypiap53_test_results['atypiap53_ground_truth_after_ai_review']))
print()
print('Atypia-P53 test Atypia-P53 triage system vs. post-AI pathologist ground truth')
print(classification_report(atypiap53_test_results['atypiap53_ground_truth_after_ai_review'], atypiap53_test_results['atypiap53_triage_scheme']))
print()
print('Atypia-P53 test Atypia-P53 triage system vs. endoscopy ground truth')
print(classification_report(atypiap53_test_results['endo_dysplasia_x'], atypiap53_test_results['atypiap53_triage_scheme']))

# Emit CSVs of these datasets
atypia_val_results.to_csv('~/Downloads/atypia_val_results.csv', index=False)
p53_val_results.to_csv('~/Downloads/p53_val_results.csv', index=False)
atypiap53_val_results.to_csv('~/Downloads/atypiap53_val_results.csv', index=False)
atypia_test_results.to_csv('~/Downloads/atypia_test_results.csv', index=False)
p53_test_results.to_csv('~/Downloads/p53_test_results.csv', index=False)
atypiap53_test_results.to_csv('~/Downloads/atypiap53_test_results.csv', index=False)
