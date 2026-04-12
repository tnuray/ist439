import tkinter as tk
import math
import numpy as np
from tkinter import messagebox

# Başlangıç dili
dil = "tr"  # "tr" = Türkçe, "en" = English

def dil_secimi():
    global dil
    dil = var_dil.get()
    guncelle_metni()

def guncelle_metni():
    if dil == "tr":
        label_baslik.config(text="Nova Bot'a Hoşgeldin! 🤖")
        buton_hesapla.config(text="Hesapla 🧮")
        buton_cikis.config(text="Çıkış 🚪")
        rb_ortalama.config(text="Ortalama")
        rb_std.config(text="Standart Sapma")
        rb_karekok.config(text="Karekök")
    else:
        label_baslik.config(text="Welcome to Nova Bot! 🤖")
        buton_hesapla.config(text="Calculate 🧮")
        buton_cikis.config(text="Exit 🚪")
        rb_ortalama.config(text="Mean")
        rb_std.config(text="Standard Deviation")
        rb_karekok.config(text="Square Root")

def hesapla():
    secim = var_islem.get()
    sonuc_label.config(text="")
    mesaj_label.config(text="")
    sayi = entry.get().split()
    try:
        sayilar = list(map(float, sayi))
        if secim == "1":  # Ortalama / Mean
            sonuc = np.mean(sayilar)
            if dil == "tr":
                sonuc_label.config(text=f"Nova: Ortalama = {sonuc:.2f} 📊")
                mesaj_label.config(text="Nova: Hadi bakalım, sayılar konuşuyor 😄")
            else:
                sonuc_label.config(text=f"Nova: Mean = {sonuc:.2f} 📊")
                mesaj_label.config(text="Nova: Let's see what the numbers say 😄")
        
        elif secim == "2":  # Standart sapma / Standard Deviation
            sonuc = np.std(sayilar, ddof=1)
            if dil == "tr":
                sonuc_label.config(text=f"Nova: Standart Sapma = {sonuc:.2f} 📈")
                mesaj_label.config(text="Nova: İstatistik ciddi olabilir ama biz eğlenceliyiz 😎")
            else:
                sonuc_label.config(text=f"Nova: Standard Deviation = {sonuc:.2f} 📈")
                mesaj_label.config(text="Nova: Statistics can be serious but we make it fun 😎")
        
        elif secim == "3":  # Kareköklü hesaplama / Square Root
            sonuc = math.sqrt(sayilar[0])
            if dil == "tr":
                sonuc_label.config(text=f"Nova: √{sayilar[0]} = {sonuc:.2f} 🔢")
                mesaj_label.config(text="Nova: Kareköklü hesaplamalar tıkır tıkır! 😁")
            else:
                sonuc_label.config(text=f"Nova: √{sayilar[0]} = {sonuc:.2f} 🔢")
                mesaj_label.config(text="Nova: Square root calculation done! 😁")
        else:
            if dil == "tr":
                sonuc_label.config(text="Nova: Lütfen geçerli bir işlem seç 😅")
            else:
                sonuc_label.config(text="Nova: Please select a valid operation 😅")
    except:
        if dil == "tr":
            sonuc_label.config(text="Nova: Hatalı giriş 😓")
            mesaj_label.config(text="Nova: Lütfen sadece sayıları doğru yaz ✨")
        else:
            sonuc_label.config(text="Nova: Invalid input 😓")
            mesaj_label.config(text="Nova: Please enter only valid numbers ✨")

def cikis():
    if dil == "tr":
        cevap = messagebox.askyesno("Çıkış", "Nova’dan çıkmak istiyor musun? 💖")
    else:
        cevap = messagebox.askyesno("Exit", "Do you want to exit Nova? 💖")
    if cevap:
        root.destroy()

# Ana pencere
root = tk.Tk()
root.title("Nova Bot 🦄")
root.geometry("500x450")
root.resizable(False, False)
root.configure(bg="#FFF8DC")

# Dil Seçimi
var_dil = tk.StringVar(value="tr")
tk.Label(root, text="Select Language / Dil Seçimi", font=("Helvetica", 10, "bold"), bg="#FFF8DC").pack(pady=5)
tk.Radiobutton(root, text="Türkçe", variable=var_dil, value="tr", command=dil_secimi, bg="#FFF8DC").pack()
tk.Radiobutton(root, text="English", variable=var_dil, value="en", command=dil_secimi, bg="#FFF8DC").pack()

# Başlık
label_baslik = tk.Label(root, text="", font=("Helvetica", 16, "bold"), bg="#FFF8DC")
label_baslik.pack(pady=10)

# İşlem Seçimi
var_islem = tk.StringVar()
rb_ortalama = tk.Radiobutton(root, text="", variable=var_islem, value="1", bg="#FFF8DC", font=("Helvetica",12))
rb_ortalama.pack(anchor="w", padx=20)
rb_std = tk.Radiobutton(root, text="", variable=var_islem, value="2", bg="#FFF8DC", font=("Helvetica",12))
rb_std.pack(anchor="w", padx=20)
rb_karekok = tk.Radiobutton(root, text="", variable=var_islem, value="3", bg="#FFF8DC", font=("Helvetica",12))
rb_karekok.pack(anchor="w", padx=20)

# Kullanıcı girişi
entry = tk.Entry(root, font=("Helvetica", 12))
entry.pack(pady=10)

# Hesapla butonu
buton_hesapla = tk.Button(root, text="", font=("Helvetica",12,"bold"), command=hesapla, bg="#FFD700")
buton_hesapla.pack(pady=5)

# Sonuç ve mesaj etiketleri
sonuc_label = tk.Label(root, text="", font=("Helvetica", 14, "bold"), bg="#FFF8DC", fg="#FF4500")
sonuc_label.pack(pady=10)
mesaj_label = tk.Label(root, text="", font=("Helvetica", 12), bg="#FFF8DC", fg="#008B8B")
mesaj_label.pack(pady=5)

# Çıkış butonu
buton_cikis = tk.Button(root, text="", font=("Helvetica",12,"bold"), command=cikis, bg="#FF6347")
buton_cikis.pack(side="bottom", pady=20)

# Dil ve metinleri başlat
guncelle_metni()

root.mainloop()
