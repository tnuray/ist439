import pandas as pd
import numpy as np
from datasets import load_dataset

dataset50 = load_dataset("yelp_review_full", split="train[:50]")

df = dataset50.to_pandas()
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




output_filename = "yelp_labeled_dataset.csv"
df.to_csv(output_filename, index=False, encoding="utf-8-sig")

print(f"Success! The file has been saved as: {output_filename}")