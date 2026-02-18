import pandas as pd # Verileri Excel gibi tablolar halinde düzenlemek ve analiz etmek için kullanılan kütüphane.
import os # Bilgisayarda klasör oluşturma ve dosya yollarını yönetme işlemlerini yapar.
import re # 'Düzenli İfadeler' (Regex) kullanarak metin içinde karmaşık arama işlemleri yapar.
from datasets import load_dataset # İnternetteki hazır veri setlerini (Hugging Face) kolayca indirmeyi sağlar.
from collections import Counter # Bir liste içindeki elemanların kaçar kez tekrar ettiğini sayan araç.

# 1. DATA LOADING # 1. VERİ YÜKLEME
print("🔄 Fetching Yelp dataset (First 50 rows)...")
# 'yelp_review_full' isimli veri setinin sadece ilk 50 satırlık eğitim (train) kısmını indiriyoruz.
dataset = load_dataset("yelp_review_full", split="train[:50]", trust_remote_code=True)

# 2. ENHANCED CATEGORY DICTIONARY # 2. KATEGORİ SÖZLÜĞÜ
# Hangi sayısal etiketin hangi anahtar kelimeleri temsil ettiğini belirliyoruz.
CATEGORIES = {
    1: ["doctor", "hospital", "dentist", "nurse", "medical", "health", "clinic", "patient", "surgery", "dr.", "specialist", "diagnosed"],# Sağlık
    2: [
        "food", "restaurant", "pizza", "burger", "dinner", "delicious", "menu", "fish", "sandwich", "sandwiches",
        "chips", "tasty", "breakfast", "meal", "sauce", "cheeseburger", "reuben", "reubens", "wings", "steak",
        "chicken", "pie", "tuna", "eat", "beer", "server", "waitress", "waiter", "bacon", "egg", "eggs",
        "cheese", "cocktails", "diner", "coffee", "lunch", "flavor", "fried"
    ],# Yemek
    3: ["car", "tire", "mechanic", "repair", "oil", "transmission", "auto", "brake", "brakes", "tires", "wheels"],# Otomobil
    5: ["plumber", "laundry", "roofing", "moving", "deck", "construction", "permit", "holes", "survey", "contractor"] # Ev Hizmetleri
}

# 3. SCORE-BASED LABELING FUNCTION # 3. ETİKETLEME FONKSİYON
def assign_numerical_label(text): #Bu fonksiyon bir metni alır, anahtar kelimelerle karşılaştırır ve enuygun kategori numarasını geri döndürür

    text = text.lower() # Büyük/küçük harf duyarlılığını kaldırmak için tüm metni küçük harfe çevirir.
    scores = Counter() # Her kategori için kaç puan toplandığını burada tutacağız.
    for label, keywords in CATEGORIES.items():
        for word in keywords:
            if re.search(rf"\b{re.escape(word)}\b", text): # \b: Kelimenin tam eşleşmesini sağlar (Örn: 'car'ı ararken 'carpet' içinde bulmaz).
                scores[label] += 1 # Eğer anahtar kelime metinde varsa o kategoriye 1 puan ekle.
    if not scores: # Eğer metinde hiçbir anahtar kelime bulunamazsa, varsayılan olarak '7' etiketini ver.
        return 7
    return scores.most_common(1)[0][0] # En yüksek puana sahip olan (en çok eşleşen) kategorinin numarasını seçer.

# 4. CREATING DATAFRAME (Fixed ID starting from 1) # 4. TABLO OLUŞTURMA
# Aşağıdaki fonksiyonu kullanarak her bir yorum metni için bir etiket oluşturuyoruz.
labels = [assign_numerical_label(t) for t in dataset["text"]]
texts = dataset["text"]

# Pandas kullanarak verileri 'ID', 'label50' ve 'text' sütunları olan bir tabloya dönüştürüyoruz.
# Burada indeksi doğrudan 1'den başlatarak oluşturuyoruz
sample50 = pd.DataFrame(
    {"label50": labels, "text": texts},
    index=range(1, len(labels) + 1)
)

# 5. SAVING FILES # 5. DOSYALARI KAYDETME
output_dir = "yelp_sample_project"
os.makedirs(output_dir, exist_ok=True) # Eğer klasör yoksa bilgisayarda yeni bir klasör oluşturur.

# index_label="ID" sütun başlığını belirler # Veriyi hem Excel ile açılabilen .csv formatında hem de profesyonel .parquet formatında kaydeder.
sample50.to_csv(os.path.join(output_dir, "sample50.csv"), index_label="ID", encoding="utf-8-sig")
sample50.to_parquet(os.path.join(output_dir, "sample50.parquet"), index=True)

# 6. CONTROL OUTPUT (80 Characters Summary)6. SONUÇLARI EKRANA YAZDIRMA (OUTPUT)
print("\n🚀 Labeling Completed!")
print(f"{'ID':<4} | {'Label50':<8} | Text Summary (80 Characters)") # Tablonun başlık kısmını hizalı bir şekilde yazdırır.
print("-" * 105)

for i, row in sample50.iterrows():
    # Artık 'i' değeri doğrudan 1, 2, 3... diye gidecek
    # Uzun metinlerin sadece ilk 80 karakterini alarak temiz bir özet görünümü oluşturur.
    clean_text = row["text"].replace("\n", " ").strip()[:80]
    print(f"{i:<4} | {row['label50']:<8} | {clean_text}...")

# Statistical Summary
# Hangi etiket kaç defa kullanılmış? (Örn: 2 numarasından 15 tane var gibi)
print("\n📊 Category Distribution:")
distribution = sample50["label50"].value_counts().sort_index()
print(distribution)



