import pandas as pd
import numpy as np
import langid
import fasttext
from datasets import load_dataset
from langdetect import detect
from langdetect import DetectorFactory

DetectorFactory.seed = 0   # aynı sonuçları almak için


#dataset50 = load_dataset("yelp_review_full", split="train[:50]")

#data_url = "https://huggingface.co/datasets/yelp_review_full/resolve/main/yelp_review_full/train-00000-of-00001.parquet"
#df_raw = pd.read_parquet(data_url).head(65000)

#df_raw['new_label_star'] = df_raw['label'] + 1

#df_raw.to_csv("yelp_stars_fixed.csv", index=False, encoding="utf-8-sig")


df = pd.read_csv("yelp_stars_fixed.csv")

df_50 = df.head(50).copy()

#df = dataset50.to_pandas()
df['new_column'] = ""

df_50.loc[0, 'new_column'] = 'health'
df_50.loc[1, 'new_column'] = 'health'
df_50.loc[2, 'new_column'] = 'health'
df_50.loc[3, 'new_column'] = 'health'
df_50.loc[4, 'new_column'] = 'health'
df_50.loc[5, 'new_column']= 'health'
df_50.loc[6, 'new_column']= 'health'
df_50.loc[7, 'new_column']= 'health'
df_50.loc[8, 'new_column']=  'food'
df_50.loc[9, 'new_column']= 'sports'
df_50.loc[10,'new_column']= 'sports'
df_50.loc[11,'new_column']= 'sports'
df_50.loc[12,'new_column']= 'sports'
df_50.loc[13, 'new_column']='other'
df_50.loc[14,'new_column']= 'food'
df_50.loc[15, 'new_column']='food'
df_50.loc[16,'new_column']='food'
df_50.loc[17,'new_column']='food'
df_50.loc[18,'new_column']='food'
df_50.loc[19,'new_column']='food'
df_50.loc[20,'new_column']='food'
df_50.loc[21,'new_column']='food'
df_50.loc[22,'new_column']='food'
df_50.loc[23, 'new_column']='food'
df_50.loc[24,'new_column']='automotive'
df_50.loc[25, 'new_column']='automotive'
df_50.loc[26, 'new_column']='automotive'
df_50.loc[27, 'new_column']='home services'
df_50.loc[28, 'new_column']='other'
df_50.loc[29,'new_column']='other'
df_50.loc[30, 'new_column']='other'
df_50.loc[31, 'new_column']='food'
df_50.loc[32, 'new_column']='food'
df_50.loc[33, 'new_column']='food'
df_50.loc[34, 'new_column']='food'
df_50.loc[35, 'new_column']='food'
df_50.loc[36, 'new_column']='food'
df_50.loc[37, 'new_column']='food'
df_50.loc[38, 'new_column']='food'
df_50.loc[39, 'new_column']='food'
df_50.loc[40, 'new_column']='food'
df_50.loc[41, 'new_column']='food'
df_50.loc[42, 'new_column']='food'
df_50.loc[43, 'new_column']='food'
df_50.loc[44, 'new_column']='food'
df_50.loc[45, 'new_column']='food'
df_50.loc[46, 'new_column']='food'
df_50.loc[47, 'new_column']='other'
df_50.loc[48, 'new_column']='food'
df_50.loc[49, 'new_column']='other'


print(df_50['new_label_star'].value_counts().sort_index())


print("Label  Distribution:")
label_counts = df_50['label'].value_counts().sort_index().to_string()
print(label_counts)
print(df_50["new_column"].value_counts())

only_fives = df_50[df_50['new_label_star'] == 5]
print(only_fives[[ 'new_label_star', 'new_column']])
print(only_fives['new_column'].value_counts().to_string())

only_fours = df_50[df_50['new_label_star'] == 4]
print(only_fours[['label', 'new_label_star', 'new_column']])
print(only_fours['new_column'].value_counts().to_string())

only_threes=df_50[df_50['new_label_star']==3]
print(only_threes[['label', 'new_label_star', 'new_column']])
print(only_threes['new_column'].value_counts().to_string())

only_twos=df_50[df_50['new_label_star']==2]
print(only_twos[['label', 'new_label_star', 'new_column']])
print(only_twos['new_column'].value_counts().to_string())

only_ones = df_50[df_50['new_label_star'] == 1]
print(only_ones[['label', 'new_label_star', 'new_column']])
print(only_ones['new_column'].value_counts())

print(df_50[df_50['new_column'] == "health"][['new_label_star', 'new_column', 'text']])
print(df_50[df_50['new_column'] == "health"]['new_label_star'].value_counts().sort_index())

output_filename = "yelp_labeled_dataset.csv"
df_50.to_csv(output_filename, index=False, encoding="utf-8-sig")





# Language Detection
# 1-LangDetect Library
def detect_language_ld(text):
    try:
        return detect(text)
    except:
        return "unknown"
df_50["language_detect"] = df_50["text"].apply(detect_language_ld)
print(df_50["language_detect"].value_counts().to_string())
non_english = df_50[df_50["language_detect"] != "en"]
print(non_english.to_string())



# 2-Langid.py Library
def detect_language_langid(text):
    try:
       lang, confidence = langid.classify(text)
       return lang
    except:
        return "unknown"
df_50["language_langid"] = df_50["text"].apply(detect_language_langid)
print(df_50["language_langid"].value_counts().to_string())

non_english_langid = df_50[df_50["language_langid"] != "en"]
print(non_english_langid.to_string())

#confidence levelları görmek istersek
#def detect_language_langid(text):
#    try:
#        return langid.classify(text)
#    except:
#       return ("unknown", 0.0)

# Veriyi işle ve sütunlara dağıt
#lang_results = df['text'].apply(detect_language_langid).tolist()
#df[['langid_code', 'langid_score']] = pd.DataFrame(lang_results, index=df.index)
#print(df[['langid_code', 'langid_score']])

# 3-FastText Library

model = fasttext.load_model("lid.176.bin")


def detect_lang(text):
    if pd.isna(text): # Eğer hücre boşsa (NaN) hata vermemesi için kontrol ekliyoruz
        return "unknown"

    text = str(text).replace("\n", " ")
    prediction = model.predict(text)
    label = prediction[0][0]
    return label.replace("__label__", "")


df_50["language_fasttext"] = df["text"].apply(detect_lang) #ilerleyen dönelmlerde tolist kullan
print(df_50["language_fasttext"].value_counts().to_string())

df_50.to_csv("yelp_processed_50_final.csv", index=False, encoding="utf-8-sig")

s1 = df[df['new_label_star'] == 1].sample(n=92, random_state=42)
s2 = df[df['new_label_star'] == 2].sample(n=82, random_state=42)
s3 = df[df['new_label_star'] == 3].sample(n=78, random_state=42)
s4 = df[df['new_label_star'] == 4].sample(n=76, random_state=42)
s5 = df[df['new_label_star'] == 5].sample(n=72, random_state=42)

final_list = pd.concat([s1, s2, s3, s4, s5])

pd.set_option('display.max_rows', None)


print(final_list['new_label_star'].value_counts().sort_index())
final_list.to_csv("yelp_400_sample.csv", index=False, encoding="utf-8-sig")

df_ag_raw = pd.read_csv("indirilen_veriler/ag_news.csv")
df_ag_50 = df_ag_raw.head(50).copy()

df_ag_50['new_column'] = ""
df_ag_50.loc[0, 'new_column'] = 'business'
df_ag_50.loc[1, 'new_column'] = 'business'
df_ag_50.loc[2, 'new_column'] = 'business'
df_ag_50.loc[3, 'new_column'] = 'business'
df_ag_50.loc[4, 'new_column']=  'business'
df_ag_50.loc[5, 'new_column']=  'business'
df_ag_50.loc[6, 'new_column']=  'business'
df_ag_50.loc[7, 'new_column']=  'business'
df_ag_50.loc[8, 'new_column']=  'business'
df_ag_50.loc[9, 'new_column']=  'business'
df_ag_50.loc[10, 'new_column']= 'business'
df_ag_50.loc[11, 'new_column']= 'business'
df_ag_50.loc[12, 'new_column']= 'business'
df_ag_50.loc[13, 'new_column']= 'business'
df_ag_50.loc[14, 'new_column']=  'business'
df_ag_50.loc[15, 'new_column']=  'business'
df_ag_50.loc[16, 'new_column']=  'business'
df_ag_50.loc[17, 'new_column']=  'business'
df_ag_50.loc[18, 'new_column']=  'business'
df_ag_50.loc[19, 'new_column']=  'business'
df_ag_50.loc[20, 'new_column']=  'business'
df_ag_50.loc[21, 'new_column']=  'business'
df_ag_50.loc[22, 'new_column']=  'business'
df_ag_50.loc[23, 'new_column']=  'business'
df_ag_50.loc[24, 'new_column']=  'business'
df_ag_50.loc[25, 'new_column']=  'business'
df_ag_50.loc[26, 'new_column']=  'business'
df_ag_50.loc[27, 'new_column']=  'business'
df_ag_50.loc[28, 'new_column']=  'business'
df_ag_50.loc[29, 'new_column']=  'business'
df_ag_50.loc[30, 'new_column']=  'business'
df_ag_50.loc[31, 'new_column']=  'business'
df_ag_50.loc[32, 'new_column']=  'business'
df_ag_50.loc[33, 'new_column']=  'business'
df_ag_50.loc[34, 'new_column']=  'business'
df_ag_50.loc[35, 'new_column']=  'business'
df_ag_50.loc[36, 'new_column']=  'business'
df_ag_50.loc[37, 'new_column']=  'business'
df_ag_50.loc[38, 'new_column']=  'business'
df_ag_50.loc[39, 'new_column']=  'business'
df_ag_50.loc[40, 'new_column']=  'business'
df_ag_50.loc[41, 'new_column']=  'business'
df_ag_50.loc[42, 'new_column']=  'business'
df_ag_50.loc[43, 'new_column']=  'business'
df_ag_50.loc[44, 'new_column']=  'business'
df_ag_50.loc[45, 'new_column']=  'business'
df_ag_50.loc[46, 'new_column']=  'business'
df_ag_50.loc[47, 'new_column']=  'business'
df_ag_50.loc[48, 'new_column']=  'business'
df_ag_50.loc[49, 'new_column']=  'business'

print(df_ag_50['new_column'].value_counts().sort_index().to_string(header=False))

print(f"Total number of records in the dataset: {len(df_ag_raw):,}")
print(df_ag_raw['label'].value_counts().sort_index().to_string(header=False))

only_threes_ag = df_ag_50[df_ag_50['label'] == 3]
print(only_threes_ag[['label', 'new_column']])
print(only_threes_ag['new_column'].value_counts().to_string())

only_twos_ag=df_ag_50[df_ag_50['label']==2]
print(only_twos_ag[['label', 'new_column']])
print(only_twos_ag['new_column'].value_counts().to_string())

only_ones_ag=df_ag_50[df_ag_50['label']==1]
print(only_ones_ag[['label', 'new_column']])
print(only_ones_ag['new_column'].value_counts().to_string())


only_zeros_ag=df_ag_50[df_ag_50['label']==0]
print(only_zeros_ag[['label', 'new_column']])
print(only_zeros_ag['new_column'].value_counts().to_string())

print(df_ag_50[df_ag_50['new_column'] == "business"][['label', 'new_column', 'text']])
print(df_ag_50[df_ag_50['new_column'] == "business"]['label'].value_counts().sort_index().to_string(header=False))

ag_output_filename = "ag_news_labeled_dataset.csv"
df_ag_50.to_csv(ag_output_filename, index=False, encoding="utf-8-sig")

# Language Detection for AG News
# 1-LangDetect Library
def detect_language_ld(text):
    try:

        return detect(text)
    except:
        return "unknown"

df_ag_50["language_detect"] = df_ag_50["text"].apply(detect_language_ld)
print(df_ag_50["language_detect"].value_counts().to_string(header=False))
non_english_ag = df_ag_50[df_ag_50["language_detect"] != "en"]



# 2-Langid.py Library

def detect_language_langid(text):
    try:
       lang, confidence = langid.classify(text)
       return lang
    except:
        return "unknown"

df_ag_50["language_langid"] = df_ag_50["text"].apply(detect_language_langid)

print(df_ag_50["language_langid"].value_counts().to_string())

non_english_langid = df_ag_50[df_ag_50["language_langid"] != "en"]

print(non_english_langid.to_string())

# 3-FastText Library
model_ft = fasttext.load_model("lid.176.bin")

def detect_lang(text):
    if pd.isna(text):
        return "unknown"

    text = str(text).replace("\n", " ")
    prediction = model_ft.predict(text)
    label = prediction[0][0]
    return label.replace("__label__", "")

df_ag_50["language_fasttext"] = df_ag_50["text"].apply(detect_lang)

print(df_ag_50["language_fasttext"].value_counts().to_string())

non_english_ft = df_ag_50[df_ag_50["language_fasttext"] != "en"]
print(non_english_ft[['label', 'language_fasttext', 'text']].to_string())

df_ag_50.to_csv("ag_news_processed_50_final.csv", index=False, encoding="utf-8-sig")

df_imdb_raw = pd.read_csv("indirilen_veriler/imdb.csv")
df_imdb_50 = df_imdb_raw.head(50).copy()

df_imdb_50['new_column'] = ""

df_imdb_50.loc[0, 'new_column'] = 'negative'
df_imdb_50.loc[1, 'new_column'] = 'negative'
df_imdb_50.loc[2, 'new_column'] = 'negative'
df_imdb_50.loc[3, 'new_column'] = 'negative'
df_imdb_50.loc[4, 'new_column'] = 'negative'
df_imdb_50.loc[5, 'new_column'] = 'negative'
df_imdb_50.loc[6, 'new_column'] = 'negative'
df_imdb_50.loc[7, 'new_column'] = 'negative'
df_imdb_50.loc[8, 'new_column'] = 'negative'
df_imdb_50.loc[9, 'new_column'] = 'negative'
df_imdb_50.loc[10, 'new_column'] = 'negative'
df_imdb_50.loc[11, 'new_column'] = 'negative'
df_imdb_50.loc[12, 'new_column'] = 'negative'
df_imdb_50.loc[13, 'new_column'] = 'negative'
df_imdb_50.loc[14, 'new_column'] = 'negative'
df_imdb_50.loc[15, 'new_column'] = 'negative'
df_imdb_50.loc[16, 'new_column'] = 'negative'
df_imdb_50.loc[17, 'new_column'] = 'negative'
df_imdb_50.loc[18, 'new_column'] = 'negative'
df_imdb_50.loc[19, 'new_column'] = 'negative'
df_imdb_50.loc[20, 'new_column'] = 'negative'
df_imdb_50.loc[21, 'new_column'] = 'negative'
df_imdb_50.loc[22, 'new_column'] = 'negative'
df_imdb_50.loc[23, 'new_column'] = 'negative'
df_imdb_50.loc[24, 'new_column'] = 'negative'
df_imdb_50.loc[25, 'new_column'] = 'negative'
df_imdb_50.loc[26, 'new_column'] = 'negative'
df_imdb_50.loc[27, 'new_column'] = 'negative'
df_imdb_50.loc[28, 'new_column'] = 'negative'
df_imdb_50.loc[29, 'new_column'] = 'negative'
df_imdb_50.loc[30, 'new_column'] = 'negative'
df_imdb_50.loc[31, 'new_column'] = 'negative'
df_imdb_50.loc[32, 'new_column'] = 'negative'
df_imdb_50.loc[33, 'new_column'] = 'negative'
df_imdb_50.loc[34, 'new_column'] = 'negative'
df_imdb_50.loc[35, 'new_column'] = 'negative'
df_imdb_50.loc[36, 'new_column'] = 'negative'
df_imdb_50.loc[37, 'new_column'] = 'negative'
df_imdb_50.loc[38, 'new_column'] = 'negative'
df_imdb_50.loc[39, 'new_column'] = 'negative'
df_imdb_50.loc[40, 'new_column'] = 'negative'
df_imdb_50.loc[41, 'new_column'] = 'negative'
df_imdb_50.loc[42, 'new_column'] = 'negative'
df_imdb_50.loc[43, 'new_column'] = 'negative'
df_imdb_50.loc[44, 'new_column'] = 'negative'
df_imdb_50.loc[45, 'new_column'] = 'negative'
df_imdb_50.loc[46, 'new_column'] = 'negative'
df_imdb_50.loc[47, 'new_column'] = 'negative'
df_imdb_50.loc[48, 'new_column'] = 'negative'
df_imdb_50.loc[49, 'new_column'] = 'negative'

print(df_imdb_50['new_column'].value_counts().sort_index().to_string(header=False))
print(f"Total number of records in the dataset: {len(df_imdb_raw):,}")
print(df_imdb_raw['label'].value_counts().sort_index().to_string(header=False))

only_zeros_imdb = df_imdb_50[df_imdb_50['label'] == 0]
print(only_zeros_imdb[['label', 'new_column']])
print(only_zeros_imdb['new_column'].value_counts().to_string())

only_ones_imdb = df_imdb_50[df_imdb_50['label'] == 1]
print(only_ones_imdb[['label', 'new_column']])
print(only_ones_imdb['new_column'].value_counts().to_string())

print(df_imdb_50[df_imdb_50['new_column'] == "negative"][['label', 'new_column', 'text']])
print(df_imdb_50[df_imdb_50['new_column'] == "negative"]['label'].value_counts().sort_index().to_string(header=False))

ag_output_filename = "ag_news_labeled_dataset.csv"
df_ag_50.to_csv(ag_output_filename, index=False, encoding="utf-8-sig")

#1-Language Detection for imdb
def detect_language_ld(text):
    try:
        return detect(text)
    except:
        return "unknown"

df_imdb_50["language_detect"] = df_imdb_50["text"].apply(detect_language_ld)


print(df_imdb_50["language_detect"].value_counts().to_string(header=False))

non_english_imdb = df_imdb_50[df_imdb_50["language_detect"] != "en"]

if non_english_imdb.empty:
    print("No non-english records found. All texts are English.")
else:
    print(non_english_imdb[['language_detect', 'text']].to_string())


#2- Langid Library (IMDB)
def detect_language_langid(text):
    try:
       lang, confidence = langid.classify(text)
       return lang
    except:
        return "unknown"

df_imdb_50["language_langid"] = df_imdb_50["text"].apply(detect_language_langid)

print(df_imdb_50["language_langid"].value_counts().to_string())

non_english_langid_imdb = df_imdb_50[df_imdb_50["language_langid"] != "en"]

print(non_english_langid_imdb.to_string())

#3- FastText Library (IMDB)

model_ft = fasttext.load_model("lid.176.bin")

def detect_lang(text):
    if pd.isna(text):
        return "unknown"

    text = str(text).replace("\n", " ")
    prediction = model_ft.predict(text)
    label = prediction[0][0]
    return label.replace("__label__", "")


df_imdb_50["language_fasttext"] = df_imdb_50["text"].apply(detect_lang)

print(df_imdb_50["language_fasttext"].value_counts().to_string())

non_english_ft_imdb = df_imdb_50[df_imdb_50["language_fasttext"] != "en"]
print(non_english_ft_imdb[['label', 'language_fasttext', 'text']].to_string())

df_imdb_50.to_csv("imdb_processed_50_final.csv", index=False, encoding="utf-8-sig")
