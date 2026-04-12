#import math
#import numpy as np

#def nova_bot():
   # print("Merhaba! Ben Nova 🤖")
    #print("Lütfen yapmak istediğin işlemi seç:")
   # print("1. Ortalama hesapla")
    #print("2. Standart sapma hesapla")
   # print("3. Kareköklü hesaplama")
    
   # secim = input("Seçimin (1/2/3): ")

   # if secim == "1":
       # sayilar = input("Sayıları aralarında boşluk bırakarak yaz: ")
       # sayilar = list(map(float, sayilar.split()))
        #ortalama = np.mean(sayilar)
        #print(f"Nova: Ortalama = {ortalama}")
    
   # elif secim == "2":
       # sayilar = input("Sayıları aralarında boşluk bırakarak yaz: ")
       # sayilar = list(map(float, sayilar.split()))
       # std = np.std(sayilar, ddof=1)  # Örneklem standart sapma
        #print(f"Nova: Standart Sapma = {std}")
    
   # elif secim == "3":
        #sayi = float(input("Karekökünü almak istediğin sayıyı yaz: "))
        #print(f"Nova: √{sayi} = {math.sqrt(sayi)}")
    
    #else:
       # print("Nova: Geçersiz seçim 😅")

# Botu çalıştır
#nova_bot()

import tkinter as tk
import math
import numpy as np
from tkinter import messagebox

def hesapla():
    secim = var.get()
    sonuc_label.config(text="")
    sayi = entry.get().split()
    
    try:
        sayilar = list(map(float, sayi))
        
        if secim == "1":
            sonuc = np.mean(sayilar)
            sonuc_label.config(text=f"Nova: Ortalama = {sonuc:.2f} 📊")
            mesaj_label.config(text="Nova: Hadi bakalım, sayılar konuşuyor 😄")
        
        elif secim == "2":
            sonuc = np.std(sayilar, ddof=1)
            sonuc_label.config(text=f"Nova: Standart Sapma = {sonuc:.2f} 📈")
            mesaj_label.config(text="Nova: İstatistik ciddiyse de biz eğlenceliyiz 😎")
        
        elif secim == "3":
            sonuc = math.sqrt(sayilar[0])
            sonuc_label.config(text=f"Nova: √{sayilar[0]} = {sonuc:.2f} 🔢")
            mesaj_label.config(text="Nova: Kareköklü hesaplamalar tıkır tıkır! 😁")
        
        else:
            sonuc_label.config(text="Nova: Lütfen geçerli bir işlem seç 😅")
            mesaj_label.config(text="")
    
    except:
        sonuc_label.config(text="Nova: Hatalı giriş 😓")
        mesaj_label.config(text="Nova: Lütfen sadece sayıları doğru yaz ✨")

def cikis():
    if messagebox.askyesno("Çıkış", "Nova’dan çıkmak istiyor musun? 💖"):
        root.destroy()

# Ana pencere
root = tk.Tk()
root.title("Nova Bot 🦄")
root.geometry("500x400")
root.resizable(False, False)
root.configure(bg="#FFF8DC")

# Başlık
tk.Label(root, text="Nova Bot'a Hoşgeldin! 🤖", font=("Helvetica", 16, "bold"), bg="#FFF8DC").pack(pady=10)

# İşlem Seçimi
var = tk.StringVar()
tk.Radiobutton(root, text="Ortalama", variable=var, value="1", bg="#FFF8DC", font=("Helvetica",12)).pack(anchor="w", padx=20)
tk.Radiobutton(root, text="Standart Sapma", variable=var, value="2", bg="#FFF8DC", font=("Helvetica",12)).pack(anchor="w", padx=20)
tk.Radiobutton(root, text="Karekök", variable=var, value="3", bg="#FFF8DC", font=("Helvetica",12)).pack(anchor="w", padx=20)

# Kullanıcı girişi
entry = tk.Entry(root, font=("Helvetica", 12))
entry.pack(pady=10)

# Hesapla butonu
tk.Button(root, text="Hesapla 🧮", font=("Helvetica",12,"bold"), command=hesapla, bg="#FFD700").pack(pady=5)

# Sonuç ve mesaj etiketleri
sonuc_label = tk.Label(root, text="", font=("Helvetica", 14, "bold"), bg="#FFF8DC", fg="#FF4500")
sonuc_label.pack(pady=10)
mesaj_label = tk.Label(root, text="", font=("Helvetica", 12), bg="#FFF8DC", fg="#008B8B")
mesaj_label.pack(pady=5)

# Çıkış butonu
tk.Button(root, text="Çıkış 🚪", font=("Helvetica",12,"bold"), command=cikis, bg="#FF6347").pack(side="bottom", pady=20)

root.mainloop()
