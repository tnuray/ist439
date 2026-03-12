import pandas as pd
import numpy as np
from datasets import load_dataset

#dataset50 = load_dataset("yelp_review_full", split="train[:50]")

data_url = "https://huggingface.co/datasets/yelp_review_full/resolve/main/yelp_review_full/train-00000-of-00001.parquet"
df = pd.read_parquet(data_url).head(50)
df['new_column'] = None

#df = dataset50.to_pandas()
df['new_column'] = np.nan

df.loc[0, 'new_column'] = 'health'
df.loc[1, 'new_column'] = 'health'
df.loc[2, 'new_column'] = 'health'
df.loc[3, 'new_column'] = 'health'
df.loc[4, 'new_column'] = 'health'
df.loc[5, 'new_column']= 'health'
df.loc[6, 'new_column']= 'health'
df.loc[7, 'new_column']= 'health'
df.loc[8, 'new_column']=  'food'
df.loc[9, 'new_column']= 'sports'
df.loc[10,'new_column']= 'sports'
df.loc[11,'new_column']= 'sports'
df.loc[12,'new_column']= 'sports'
df.loc[13, 'new_column']='other'
df.loc[14,'new_column']= 'food'
df.loc[15, 'new_column']='food'
df.loc[16,'new_column']='food'
df.loc[17,'new_column']='food'
df.loc[18,'new_column']='food'
df.loc[19,'new_column']='food'
df.loc[20,'new_column']='food'
df.loc[21,'new_column']='food'
df.loc[22,'new_column']='food'
df.loc[23, 'new_column']='food'
df.loc[24,'new_column']='automotive'
df.loc[25, 'new_column']='automotive'
df.loc[26, 'new_column']='automotive'
df.loc[27, 'new_column']='home services'
df.loc[28, 'new_column']='other'
df.loc[29,'new_column']='other'
df.loc[30, 'new_column']='other'
df.loc[31, 'new_column']='food'
df.loc[32, 'new_column']='food'
df.loc[33, 'new_column']='food'
df.loc[34, 'new_column']='food'
df.loc[35, 'new_column']='food'
df.loc[36, 'new_column']='food'
df.loc[37, 'new_column']='food'
df.loc[38, 'new_column']='food'
df.loc[39, 'new_column']='food'
df.loc[40, 'new_column']='food'
df.loc[41, 'new_column']='food'
df.loc[42, 'new_column']='food'
df.loc[43, 'new_column']='food'
df.loc[44, 'new_column']='food'
df.loc[45, 'new_column']='food'
df.loc[46, 'new_column']='food'
df.loc[47, 'new_column']='other'
df.loc[48, 'new_column']='food'
df.loc[49, 'new_column']='other'

print("Label  Distribution:")
label_counts = df['label'].value_counts().sort_index().to_string()
print(label_counts)
print(df["new_column"].value_counts())

only_fours = df[df['label'] == 4]
print(only_fours)
print(only_fours['new_column'].value_counts().to_string())

only_threes = df[df['label'] == 3]
print(only_threes)
print(only_threes['new_column'].value_counts().to_string())

only_twos=df[df['label']==2]
print(only_twos)
print(only_twos['new_column'].value_counts().to_string())

only_ones=df[df['label']==1]
print(only_ones)
print(only_ones['new_column'].value_counts().to_string())

only_zeros = df[df['label'] == 0]
print(only_zeros)
print(only_zeros['new_column'].value_counts())

print(df[df['new_column'] == "health"])
print(df[df['new_column'] == "health"]['label'].value_counts().sort_index())

output_filename = "yelp_labeled_dataset.csv"
df.to_csv(output_filename, index=False, encoding="utf-8-sig")

