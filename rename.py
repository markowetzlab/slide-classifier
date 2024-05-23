import os
import pandas as pd

#Read in csv file, replace '' with file
df = pd.read_csv('results/BEST4/ViT-L/pilot/he/pilot-triage-atypia-prediction-data.csv')
#replace with path to your files

path_to_files = 'results/BEST4/ViT-L/pilot/he/inference'

#print first few columns to check
df.head()

#replace string using value from second column
#col1 is path to files, col2 is new file names
col1 = df['Case Path'].to_list()
col2 = df['CYT ID'].to_list()

extension = '_inference.pml'

for x, y in zip(col1, col2):
    file_to_rename = os.path.join(path_to_files, x)
    file_to_rename = file_to_rename.replace('.svs', extension)
    if os.path.exists(file_to_rename):
        x = file_to_rename.replace(extension, '')
        print(f'Renaming {x} to {y}')
        new_file_name = file_to_rename.replace(x, y)
        os.rename(file_to_rename, new_file_name)