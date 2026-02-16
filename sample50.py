import pandas as pd
import os
import re
from datasets import load_dataset

# 1. VERİYİ ÇEKME
print("🔄 Yelp veri seti çekiliyor (İlk 50 satır)...")
dataset = load_dataset("yelp_review_full", split="train[:50]", trust_remote_code=True)


# 2. TAM KELİME EŞLEŞME FONKSİYONU
def kelime_var_mi(text, kelimeler):
    # Kelimeleri metin içinde tek tek arar, tam eşleşme (\b) kontrolü yapar
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in kelimeler)


# 3. SAYISAL LABEL ATAMA FONKSİYONU
def sayisal_label_atayi(text):
    text = text.lower()

    # --- ÖNCELİKLİ KATEGORİLER ---

    # KATEGORİ 1: Sağlık ve Tıp (Doktor, Hastane, Dişçi vb.)
    if kelime_var_mi(text, ["doctor", "hospital", "dentist", "nurse", "medical", "health", "clinic", "practitioner",
                            "diagnosed", "patient"]):
        return 1

    # KATEGORİ 3: Otomotiv Hizmetleri (Araba, Lastik, Tamir, Fren vb.)
    elif kelime_var_mi(text, ["car", "tire", "mechanic", "repair", "oil", "transmission", "auto", "brake", "brakes"]):
        return 3

    # KATEGORİ 4: Kişisel Bakım ve Güzellik (Saç, Salon, Spa, Berber vb.)
    elif kelime_var_mi(text, ["hair", "salon", "nails", "spa", "barber", "waxing"]):
        return 4

    # KATEGORİ 6: Konaklama ve Seyahat (Otel, Oda, Tatil, Yolculuk vb.)
    elif kelime_var_mi(text, ["hotel", "stay", "room", "weekend", "trip", "travel", "inn", "hostel"]):
        return 6

    # KATEGORİ 2: Restoran ve Yiyecek (Yemek, Pizza, Balık, Sandviç, Kahvaltı vb.)
    elif kelime_var_mi(text,
                       ["food", "restaurant", "pizza", "burger", "dinner", "delicious", "menu", "fish", "sandwich",
                        "chips", "tasty", "breakfast", "lunch", "diner", "meal", "sauce", "cheeseburger", "reuben"]):
        return 2

    # KATEGORİ 5: Ev ve Yerel Hizmetler (Tesisat, Temizlik, Çatı, Nakliye vb.)
    elif kelime_var_mi(text, ["clean", "plumber", "laundry", "roofing", "moving", "deck"]):
        return 5

    # KATEGORİ 7: Diğer Hizmetler (Mağazalar, Spor Alanları ve Tanımlanamayanlar)
    else:
        return 7


# 4. 'sample50' DATAFRAME OLUŞTURMA
# Burada tablo ismini 'sample50'  koda işliyoruz.
sample50 = pd.DataFrame()
sample50["label"] = [sayisal_label_atayi(t) for t in dataset["text"]]
sample50["text"] = dataset["text"]

# 5. DOSYALARI 'sample50' ADIYLA KAYDETME
output_dir = "yelp_sample_project"
os.makedirs(output_dir, exist_ok=True)

# Hem CSV hem Parquet formatında kaydediliyor
sample50.to_csv(os.path.join(output_dir, "sample50.csv"), index=False, encoding="utf-8-sig")
sample50.to_parquet(os.path.join(output_dir, "sample50.parquet"), index=False)

# 6. EKRAN ÇIKTISI
print("\n" + "=" * 30)
print("🚀 SAMPLE 50 TABLOSU HAZIR")
print("=" * 30)
print(f"{'label':<6} | text")
print("-" * 85)

for _, row in sample50.iterrows():
    # Metni temizle ve ilk 90 karakterini göster
    temiz_metin = row["text"].replace("\n", " ").strip()[:90]
    print(f"{row['label']:<6} | {temiz_metin}...")

print("\n✅ Veriler etiketlendi ve 'yelp_sample_project/sample50.csv' olarak kaydedildi.")


