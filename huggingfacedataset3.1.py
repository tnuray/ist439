import os    # Bilgisayarın dosya ve klasör sistemine (klasör açma, yol bulma) ulaşmak için.
import pandas as pd  # Verileri tablolar halinde işlemek ve farklı formatlarda kaydetmek için.
from datasets import load_dataset # Hugging Face'ten hazır veri setlerini indiren ana araç.

# İndirmek istediğimiz veri setlerinin isimlerini bir liste içinde topluyoruz.İsteğe bağlı yeni veri setleri de eklenebilir 6.satırdaki listeye.
dataset_names = ["imdb", "ag_news", "yelp_review_full"]
output_dir = "indirilen_veriler" # Verilerin kaydedileceği klasörün adını belirliyoruz.

# CSV kaydı istiyor musun? True = Evet, False = Hayır , True ise hem Parquet hem CSV kaydeder, False ise sadece Parquet.
SAVE_CSV = True

os.makedirs(output_dir, exist_ok=True) # Klasörü oluşturur. exist_ok=True, klasör zaten varsa hata vermemesini sağlar.

for dataset_name in dataset_names: # Listedeki her bir veri seti ismi için sırayla işlem yap.
    try:  # Hata çıkma ihtimaline karşı bu bloğu "dene", hata olursa çökme.
        print(f"--- {dataset_name} İşleniyor ---")

        dataset = load_dataset(dataset_name, trust_remote_code=True) # Veriyi Hugging Face'ten çek. trust_remote_code=True güvenli yükleme sağlar.

        split = 'train' if 'train' in dataset else list(dataset.keys())[0] # 'train' (eğitim) bölümü varsa onu seç, yoksa mevcut olan ilk bölümü al (çökmemek için).
        df = dataset[split].to_pandas() # Hugging Face formatındaki veriyi Pandas'ın tablo (DataFrame) formatına çevir.

        # 1. Her zaman Parquet olarak kaydet (Verimli olan)
        parquet_path = os.path.join(output_dir, f"{dataset_name}.parquet")# Parquet Kaydı: Verimli ve hızlı format için dosya yolunu oluştur ve kaydet.
        df.to_parquet(parquet_path, index=False)
        print(f"✅ Parquet kaydedildi.")

        # 2. Eğer SAVE_CSV True ise CSV olarak da kaydet
        if SAVE_CSV:
            csv_path = os.path.join(output_dir, f"{dataset_name}.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig") # utf-8-sig: Türkçe karakterlerin Excel'de bozulmadan görünmesini sağlar.
            print(f"✅ CSV kaydedildi.")

        print(f"📊 Toplam Satır: {len(df)}\n") # İşlem bittiğinde o veri setinde kaç satır olduğunu ekrana yazdır.

    except Exception as e:
        print(f" [HATA] {dataset_name} indirilirken bir sorun oluştu: {e}")# Eğer yukarıdaki adımlarda bir hata oluşursa, hatayı yazdır ve sıradakine geç.
        continue

print("\nTüm işlemler tamamlandı.")