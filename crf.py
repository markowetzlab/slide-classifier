import pandas as pd

# Path to the directory containing the images
qc_path = '/media/prew01/BEST/BEST4/surveillance/data/he/40x_400/qc/process_list.csv'
atypia_path = '/media/prew01/BEST/BEST4/surveillance/data/he/40x_400/process_list.csv'
p53_path = '/media/prew01/BEST/BEST4/surveillance/data/p53/40x_400/process_list.csv'
tff3_path = '/media/prew01/BEST/BEST4/surveillance/data/tff3/40x_400/process_list.csv'

print(f'Loading data from \n{qc_path} \n{atypia_path} \n{p53_path} \n{tff3_path}')

# Load the data
qc = pd.read_csv(qc_path)
#Remap Column Names
qc_column_mapping = {
    'slide_filename': 'h_e_slide_filename',
    'positive_tiles': 'he_numb_gastric_tile_alg',
    'algorithm_result': 'h_e_qc_result_alg',
    'tile_mapping': 'qc_tile_mapping',
    'algorithm_version': 'qc_algorithm_version'
}
qc = qc.rename(columns=qc_column_mapping)

atypia = pd.read_csv(atypia_path)
atypia_column_mapping = {
    'slide_filename': 'h_e_slide_filename',
    'positive_tiles': 'atypia_positive_tiles',
    'algorithm_result': 'atypia_algorithm_result',
    'tile_mapping': 'atypia_tile_mapping',
    'algorithm_version': 'atypia_algorithm_version'
}
atypia = atypia.rename(columns=atypia_column_mapping)

p53 = pd.read_csv(p53_path)
p53_column_mapping = {
    'slide_filename': 'p53_slide_filename',
    'positive_tiles': 'p53_positive_tiles',
    'algorithm_result': 'p53_algorithm_result',
    'tile_mapping': 'p53_tile_mapping',
    'algorithm_version': 'p53_algorithm_version'
}
p53 = p53.rename(columns=p53_column_mapping)

tff3 = pd.read_csv(tff3_path)
tff3_column_mapping = {
    'slide_filename': 'tff3_slide_filename',
    'positive_tiles': 'tff3_positive_tiles',
    'algorithm_result': 'tf3_algorithm_result',
    'tile_mapping': 'tff3_tile_mapping',
    'algorithm_version': 'tff3_algorithm_version'
}
tff3 = tff3.rename(columns=tff3_column_mapping)

# Step 1: Initialize DataFrame for mapping ids
record_ids = pd.read_csv('/media/prew01/BEST/BEST4/BarrettsOESophagusTr-InformationForMachin_DATA_2024-07-10_1629.csv')
record_ids = record_ids.dropna(subset=['cypath_lab_nmb'])
record_ids['redcap_event_name'] = 'unscheduled_arm_1'
record_ids['redcap_repeat_instrument'] = 'machine_learning_pathology_results'
record_ids['redcap_repeat_instance'] = 1

appended_df = pd.DataFrame(columns=['record_id', 'redcap_event_name', 'redcap_repeat_instrument', 'redcap_repeat_instance', 'redcap_data_access_group'])

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
for index, row in appended_df.iterrows():
    if row['algorithm_cyted_sample_id'] in record_ids['cypath_lab_nmb'].values:
        matching_row = record_ids[record_ids['cypath_lab_nmb'] == row['algorithm_cyted_sample_id']]
    elif row['algorithm_cyted_sample_id'] in record_ids['cypath_lab_nmb_rep'].values:
        matching_row = record_ids[record_ids['cypath_lab_nmb_rep'] == row['algorithm_cyted_sample_id']]
        matching_row.loc[matching_row.index, 'redcap_repeat_instance'] = 2
    else:
        continue
    appended_df.loc[index, appended_df.columns[:5]] = matching_row.iloc[0, :5]

# Step 5: Reorder the columns
record_id = appended_df.pop('record_id')
repeat = appended_df.pop('redcap_repeat_instance')

# Insert the record_id and redcap_repeat_instance columns at the beginning of the DataFrame
appended_df.insert(0, 'record_id', record_id)
appended_df.insert(1, 'redcap_repeat_instance', repeat)

# Sort the DataFrame by record_id and redcap_repeat_instance
appended_df.sort_values(by=['record_id', 'redcap_repeat_instance'], inplace=True)

output_path = '/media/prew01/BEST/BEST4/surveillance/BEST4_AI_crfs.csv'
print(f'Saving appended data to {output_path}')
appended_df.to_csv(output_path, index=False)  # Save the appended data to a CSV file


