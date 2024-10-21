import os
import time
import pandas as pd
import shutil

date = time.strftime('%y%m07')

# Path to the directory containing the results
base_path = '/media/prew01/BEST/BEST4/surveillance/'
# Output directory for the combined results
output_dir = '/media/prew01/BEST/BEST4/surveillance/results/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Load the data
qc = pd.read_csv(os.path.join(base_path, f'he/features/40x_400/results/qc_process_list_{date}.csv'))
#Remap Column Names
qc_column_mapping = {
    'slide_filename': 'h_e_slide_filename',
    'positive_tiles': 'he_numb_gastric_tile_alg',
    'algorithm_result': 'h_e_qc_result_alg',
    'tile_mapping': 'qc_tile_mapping',
    'algorithm_version': 'qc_algorithm_version'
}
qc = qc.rename(columns=qc_column_mapping)

atypia = pd.read_csv(os.path.join(base_path, f'he/features/40x_400/results/he_process_list_{date}.csv'))
atypia_column_mapping = {
    'slide_filename': 'h_e_slide_filename',
    'positive_tiles': 'atypia_positive_tiles',
    'algorithm_result': 'atypia_algorithm_result',
    'tile_mapping': 'atypia_tile_mapping',
    'algorithm_version': 'atypia_algorithm_version'
}
atypia = atypia.rename(columns=atypia_column_mapping)

p53 = pd.read_csv(os.path.join(base_path, f'p53/features/40x_400/results/p53_process_list_{date}.csv'))
p53_column_mapping = {
    'slide_filename': 'p53_slide_filename',
    'positive_tiles': 'p53_positive_tiles',
    'algorithm_result': 'p53_algorithm_result',
    'tile_mapping': 'p53_tile_mapping',
    'algorithm_version': 'p53_algorithm_version'
}
p53 = p53.rename(columns=p53_column_mapping)

tff3 = pd.read_csv(os.path.join(base_path, f'tff3/features/40x_400/results/tff3_process_list_{date}.csv'))
tff3_column_mapping = {
    'slide_filename': 'tff3_slide_filename',
    'positive_tiles': 'tff3_positive_tiles',
    'algorithm_result': 'tf3_algorithm_result',
    'tile_mapping': 'tff3_tile_mapping',
    'algorithm_version': 'tff3_algorithm_version'
}
tff3 = tff3.rename(columns=tff3_column_mapping)

# Step 1: Initialize DataFrame for mapping ids
record_ids = pd.read_csv(os.path.join(base_path, 'data/BarrettsOESophagusTr-WillGretaSpongePatho_DATA_LABELS_2024-10-08_0810.csv'))
record_ids = record_ids.dropna(subset=['Cyted Lab Number (Format: YYCYT#####)'])
participant_ids = dict(zip(record_ids['Cyted Lab Number (Format: YYCYT#####)'], record_ids['Participant ID:  ']))

repeat_record_ids = record_ids.dropna(subset=['Cyted Lab Number (Format: YYCYT#####).1'])
repeat_participant_ids = dict(zip(repeat_record_ids['Cyted Lab Number (Format: YYCYT#####).1'], repeat_record_ids['Participant ID:  ']))

appended_df = pd.DataFrame(columns=['record_id', 'redcap_event_name', 'redcap_repeat_instrument', 'redcap_repeat_instance'])

# Step 2: List of DataFrames to append
dfs = [qc, atypia, p53, tff3]

# Step 3: Iterate through DataFrames and append only unique columns
for df in dfs:
    # Identify unique columns in the current DataFrame that are not in appended_df
    unique_cols = [col for col in df.columns if col not in appended_df.columns]
    # Select only the unique columns for appending
    df_unique = df[unique_cols]
    # Append the DataFrame with only the unique columns
    appended_df = pd.concat([appended_df, df_unique], axis=1)

# Step 4: Iterate through the appended_df and record_ids to match the record_id
not_found_cases = []
for index, row in appended_df.iterrows():
    cyted_sample_id = row['algorithm_cyted_sample_id']
    if cyted_sample_id in participant_ids.keys():
        appended_df.at[index, 'record_id'] = participant_ids[cyted_sample_id]
        appended_df.at[index, 'redcap_repeat_instance'] = 1
    elif cyted_sample_id in repeat_participant_ids.keys():
        appended_df.at[index, 'record_id'] = repeat_participant_ids[cyted_sample_id]
        appended_df.at[index, 'redcap_repeat_instance'] = 2
    else:
        not_found_cases.append(cyted_sample_id)

# Print cases not found in the reference dataset
if not_found_cases:
    print("Cases not found in the dataset:")
    for case in not_found_cases:
        print(case)

appended_df['redcap_event_name'] = 'unscheduled_arm_1'
appended_df['redcap_repeat_instrument'] = 'machine_learning_pathology_results'

# Sort the DataFrame by record_id and redcap_repeat_instance
appended_df.sort_values(by=['record_id', 'redcap_repeat_instance'], inplace=True)

output_path = os.path.join(output_dir, f'BEST4_AI_crfs_{date}.csv')
print(f'Saving appended data to {output_dir}')
appended_df.to_csv(output_path, index=False)  # Save the appended data to a CSV file

for case, row in appended_df.iterrows():
    best4_case_id = row['record_id']
    if pd.isnull(best4_case_id):
        continue
    repeat = row["redcap_repeat_instance"]
    instance = f'{best4_case_id}-{repeat}'
    case_dir = os.path.join(output_dir, instance)
    os.makedirs(case_dir, exist_ok=True)

    # Save the individual results to the case directory
    shutil.copytree(os.path.join(base_path, f'he/features/40x_400/results/{row["h_e_slide_filename"]}'), f'{case_dir}/{row["h_e_slide_filename"]}')
    shutil.copytree(os.path.join(base_path, f'p53/features/40x_400/results/{row["p53_slide_filename"]}'), f'{case_dir}/{row["p53_slide_filename"]}')
    shutil.copytree(os.path.join(base_path, f'tff3/features/40x_400/results/{row["tff3_slide_filename"]}'), f'{case_dir}/{row["tff3_slide_filename"]}')

    #zip the case directory and save to the output directory and delete the case directory
    shutil.make_archive(os.path.join(output_dir, instance), 'zip', os.path.join(output_dir, instance))
    shutil.rmtree(case_dir)

