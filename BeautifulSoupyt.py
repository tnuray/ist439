import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time  # Engel yememek için bekleme ekliyoruz
import random  # Rastgele süreler için
from colorama import init, Fore, Style

init(autoreset=True)

# 1. HATA ÇÖZÜMÜ: Headers sözlük yapısına getirildi ve zenginleştirildi
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.google.com/"  # Google üzerinden gelmiş gibi yapıyoruz
}

url1 = "https://dergipark.org.tr/tr/pub/sdufenbed/article/1511196"


def get_data(url):
    try:
        # 2. ENGEL ÇÖZÜMÜ: İnsan gibi davranmak için rastgele bekleme (3-6 saniye)
        time.sleep(random.uniform(3, 6))

        # Oturum (session) kullanarak çerezleri yönetiyoruz
        session = requests.Session()
        r = session.get(url, headers=headers, timeout=20)

        if r.status_code == 200:
            return BeautifulSoup(r.content, "html.parser")
        else:
            print(Fore.RED + f"Erişim Engellendi! Hata Kodu: {r.status_code}")
            return None
    except Exception as e:
        print(Fore.RED + f"Bağlantı hatası: {e}")
        return None


# Veri çekme işlemi başlatılıyor
soup = get_data(url1)

if soup:
    # 3. ÇÖKME ÇÖZÜMÜ: 'NoneType' hatasını engellemek için if kontrolleri eklendi

    # Başlık çekme
    baslik_obj = soup.find("h3", class_="article-title")
    baslik = baslik_obj.text.strip() if baslik_obj else "Başlık Bulunamadı"

    # Künye bilgisi
    alt_bilgi_obj = soup.find("span", class_="article-subtitle")
    alt_bilgi = alt_bilgi_obj.text.strip() if alt_bilgi_obj else "Künye Bilgisi Yok"

    # PDF linki çekme (Güvenli kontrol)
    pdf_butonu = soup.find("a", class_="pdf")
    if pdf_butonu:
        pdf_linki = pdf_butonu.get("href")
        if pdf_linki and pdf_linki.startswith("/"):
            pdf_linki = "https://dergipark.org.tr" + pdf_linki
    else:
        pdf_linki = "PDF Linki Mevcut Değil"

    # Terminale Yazdırma (Renklendirme)
    print(Fore.CYAN + "=" * 50)
    print(Fore.GREEN + "MAKALE ADI : " + Style.RESET_ALL + baslik)
    print(Fore.YELLOW + "BİLGİLER    : " + alt_bilgi)
    print(Fore.MAGENTA + "PDF LİNKİ   : " + Style.BRIGHT + pdf_linki)
    print(Fore.CYAN + "=" * 50)

    # Kaynakça (find.all -> find_all hatası düzeltildi)
    kaynakca_alani = soup.find("div", class_="article-citations")
    if kaynakca_alani:
        kaynaklar = kaynakca_alani.find_all("li")
        for i, kaynak in enumerate(kaynaklar, 1):
            print(Fore.WHITE + f"{i}. {kaynak.text.strip()}")
    else:
        print(Fore.RED + "Kaynakça bulunamadı.")
    print(Fore.CYAN + "=" * 50)

    # 4. EXCEL KAYIT (Sözlükteki virgül ve parantez hataları düzeltildi)
    makale_verisi = [{
        "Makale Başlığı": baslik,
        "Künye": alt_bilgi,
        "PDF Adresi": pdf_linki,
        "İşlem Tarihi": datetime.now().strftime("%d-%m-%Y %H:%M")
    }]

    df = pd.DataFrame(makale_verisi)
    df.to_excel("Makale_Listesi.xlsx", index=False)

    print(Fore.LIGHTGREEN_EX + Style.BRIGHT + "\nBAŞARILI: Veriler Excel dosyasına kaydedildi.")

else:
    print(Fore.RED + "İşlem başarısız. Site içeriği alınamadı.")