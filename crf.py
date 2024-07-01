import pandas as pd

# Path to the directory containing the images
qc_path = '/media/prew01/BEST/BEST4/surveillance/he/inference/qc/process_list.csv'
atypia_path = '/media/prew01/BEST/BEST4/surveillance/he/inference/process_list.csv'
p53_path = '/media/prew01/BEST/BEST4/surveillance/p53/inference/process_list.csv'
tff3_path = '/media/prew01/BEST/BEST4/surveillance/tff3/inference/process_list.csv'

# Load the data
qc = pd.read_csv(qc_path, index_col=0)

#Remap Column Names
# Example of a mapping dictionary - update this according to your needs
qc_column_mapping = {
    'slide_filename': 'h_e_slide_filename',
    'positive_tiles': 'he_numb_gastric_tile_alg',
    'algorithm_result': 'h_e_qc_result_alg',
    'tile_mapping': 'qc_tile_mapping',
    'algorithm_version': 'qc_algorithm_version'
}
qc = qc.rename(columns=qc_column_mapping)

atypia = pd.read_csv(atypia_path, index_col=0)
atypia_column_mapping = {
    'slide_filename': 'h_e_slide_filename',
    'positive_tiles': 'atypia_positive_tiles',
    'algorithm_result': 'atypia_algorithm_result',
    'tile_mapping': 'atypia_tile_mapping',
    'algorithm_version': 'atypia_algorithm_version'
}
atypia = atypia.rename(columns=atypia_column_mapping)

p53 = pd.read_csv(p53_path, index_col=0)
p53_column_mapping = {
    'slide_filename': 'p53_slide_filename',
    'positive_tiles': 'p53_positive_tiles',
    'algorithm_result': 'p53_algorithm_result',
    'tile_mapping': 'p53_tile_mapping',
    'algorithm_version': 'p53_algorithm_version'
}
p53 = p53.rename(columns=p53_column_mapping)

tff3 = pd.read_csv(tff3_path, index_col=0)
tff3_column_mapping = {
    'slide_filename': 'tff3_slide_filename',
    'positive_tiles': 'tff3_positive_tiles',
    'algorithm_result': 'tf3_algorithm_result',
    'tile_mapping': 'tff3_tile_mapping',
    'algorithm_version': 'tff3_algorithm_version'
}
tff3 = tff3.rename(columns=tff3_column_mapping)

# Step 1: Initialize an empty DataFrame
appended_df = pd.DataFrame()

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

# column_order = []
# appended_df = appended_df[column_order]
output_path = '/media/prew01/BEST/BEST4/surveillance/BEST4_AI_crfs.csv'
appended_df.to_csv(output_path)  # Save the appended data to a CSV file


