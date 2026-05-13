import customtkinter as ctk
import pandas as pd
import numpy as np
import os
from tkinter import messagebox, filedialog
from tabulate import tabulate
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

model_ft = fasttext.load_model("lid.176.bin")


def detect_language_ft(text):
    if pd.isna(text):
        return "unknown"
    text = str(text).replace("\n", " ")
    prediction = model_ft.predict(text)
    return prediction[0][0].replace("__label__", "")


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ── Sampling Functions ───────

def sample_yelp():
    df = pd.read_csv("yelp_stars_fixed.csv")
    s1 = df[df['new_label_star'] == 1].sample(n=92)
    s2 = df[df['new_label_star'] == 2].sample(n=82)
    s3 = df[df['new_label_star'] == 3].sample(n=78)
    s4 = df[df['new_label_star'] == 4].sample(n=76)
    s5 = df[df['new_label_star'] == 5].sample(n=72)
    result = pd.concat([s1, s2, s3, s4, s5]).reset_index(drop=True)
    return result, "new_label_star"


def sample_ag_news():
    df = pd.read_csv("indirilen_veriler/ag_news.csv")
    df['row_number'] = range(1, len(df) + 1)
    k = 313
    start_row = np.random.randint(1, k + 1)
    print(f"[AG News] Selected starting row number: {start_row}")
    result = df.iloc[(start_row - 1)::k].copy().reset_index(drop=True)
    return result, "label"


def sample_imdb():
    df = pd.read_csv("indirilen_veriler/imdb.csv")
    df['row_number'] = range(1, len(df) + 1)
    k = 65
    start_row = np.random.randint(1, k + 1)
    print(f"[IMDB] Selected starting row number: {start_row}")
    result = df.iloc[(start_row - 1)::k].copy().reset_index(drop=True)
    return result, "label"


# ── GUI ─────

class DataExplor(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Data Explorer 🦄")
        self.geometry("950x800")

        self.current_df = None
        self.current_filename = None

        # Title
        ctk.CTkLabel(self, text="Data Explorer Bot 🤖",
                     font=("Helvetica", 24, "bold")).pack(pady=20)
        ctk.CTkLabel(self, text="Select a dataset to generate a new random sample:",
                     font=("Helvetica", 14)).pack(pady=5)

        # Dataset Buttons
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=15)

        ctk.CTkButton(self.button_frame, text="Yelp Reviews",
                      command=lambda: self.load_sample("yelp")).grid(
            row=0, column=0, padx=10, pady=10)
        ctk.CTkButton(self.button_frame, text="AG News",
                      command=lambda: self.load_sample("ag_news")).grid(
            row=0, column=1, padx=10, pady=10)
        ctk.CTkButton(self.button_frame, text="IMDB Reviews",
                      command=lambda: self.load_sample("imdb")).grid(
            row=0, column=2, padx=10, pady=10)

        # Display Area
        self.text_display = ctk.CTkTextbox(self, width=900, height=450,
                                           font=("Consolas", 12))
        self.text_display.pack(pady=20, padx=20)
        self.text_display.insert("0.0",
                                 "System: Ready. Please select a dataset to generate a sample... ✨\n")

        # Save Button (bottom left)
        self.save_button = ctk.CTkButton(self, text="💾 Save (CSV)",
                                         command=self.save_csv,
                                         state="disabled", fg_color="gray")
        self.save_button.pack(pady=5, side="left", padx=(20, 5))

        # Summary Button (bottom right)
        self.summary_button = ctk.CTkButton(self, text="📊 Summary Statistics",
                                            command=self.show_summary,
                                            state="disabled", fg_color="gray")
        self.summary_button.pack(pady=5, side="left", padx=5)

        # sentiment buton
        self.predict_button = ctk.CTkButton(self, text="🤖 Sentiment Prediction",
                                            command=self.run_llm_prediction)
        self.predict_button.pack(pady=5, side="left", padx=5)

        # Status Bar
        self.status_label = ctk.CTkLabel(self, text="Status: Idle",
                                         font=("Helvetica", 12), text_color="gray")
        self.status_label.pack(side="bottom", pady=10)

    def load_sample(self, dataset):
        try:
            self.status_label.configure(text="Status: Sampling...", text_color="orange")
            self.update()

            if dataset == "yelp":
                df, label_col = sample_yelp()
                filename = "yelp"
            elif dataset == "ag_news":
                df, label_col = sample_ag_news()
                filename = "ag_news"
            elif dataset=="imdb":
                df, label_col = sample_imdb()
                filename = "imdb"
            else:
                return



            self.current_df = df
            self.current_filename = filename

            cols = [c for c in [label_col, 'new_column', 'text'] if c in df.columns]
            display_df = df[cols].copy()
            display_df['text'] = display_df['text'].astype(str).str.slice(0, 70) + "..."

            table = tabulate(display_df, headers='keys', tablefmt='psql', showindex=True)

            self.text_display.delete("1.0", "end")
            self.text_display.insert("end",
                                     f"--- New Sample: {filename} (n={len(df)}) ---\n\n")
            self.text_display.insert("end", table)
            self.text_display.insert("end",
                                     f"\n\n--- Label Distribution ({label_col}) ---\n")
            self.text_display.insert("end",
                                     df[label_col].value_counts().sort_index().to_string())

            self.status_label.configure(text="Status: Detecting languages...", text_color="orange")
            self.update()

            lang_counts = df['text'].apply(detect_language_ft).value_counts()
            total = len(df)
            lang_summary = ", ".join([f"{lang}: %{count / total * 100:.1f}" for lang, count in lang_counts.items()])

            self.save_button.configure(state="normal", fg_color="#1f6aa5")
            self.summary_button.configure(state="normal", fg_color="#1f6aa5")
            if dataset == "ag_news":
                self.predict_button.configure(command=self.run_ag_news_prediction)
            elif  dataset=="imdb":
                self.predict_button.configure(command=self.run_imdb_prediction)
            else:
                self.predict_button.configure(command=self.run_llm_prediction)
            self.status_label.configure(
                text=f"{filename} ({len(df)} rows) | {lang_summary}",
                text_color="#00FF00")

        except FileNotFoundError as e:
            messagebox.showerror("File Not Found!", f"Raw CSV file not found:\n{e}")
            self.status_label.configure(text="Status: File not found!", text_color="red")
        except Exception as e:
            messagebox.showerror("Error!", f"An unexpected error occurred:\n{str(e)}")
            self.status_label.configure(text="Status: Error.", text_color="red")

    def save_csv(self):
        if self.current_df is None:
            messagebox.showwarning("Warning", "No sample to save.")
            return

        answer = messagebox.askyesno(
            "Save",
            f"Do you want to save the '{self.current_filename}' sample as a CSV file?"
        )

        if answer:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile=f"{self.current_filename}_sample.csv"
            )
            if filepath:
                self.current_df.to_csv(filepath, index=False, encoding="utf-8-sig")
                self.status_label.configure(
                    text=f"Status: Saved → {filepath}", text_color="#00FF00")
                messagebox.showinfo("Succeeded!", f"File saved successfully:\n{filepath}")

    def show_summary(self):
        # Güvenli kontrol
        if self.current_df is None: return

        # 1. ÖNCE GENEL İSTATİSTİKLER GELSİN
        raw_info = {
            "yelp": ("yelp_stars_fixed.csv", "new_label_star"),
            "ag_news": ("indirilen_veriler/ag_news.csv", "label"),
            "imdb": ("indirilen_veriler/imdb.csv", "label")
        }

        try:
            path, label_col = raw_info[self.current_filename]
            df_raw = pd.read_csv(path)

            summary = f"\n--- Full Dataset Summary: {self.current_filename} ---\n"
            summary += f"Total rows: {len(df_raw):,}\n\n"
            summary += f"Label Distribution:\n{df_raw[label_col].value_counts().sort_index().to_string()}\n"
            summary += "=" * 65 + "\n"

            self.text_display.insert("end", summary)
        except Exception as e:
            self.text_display.insert("end", f"\n[Error] Statistics could not be loaded: {e}\n")

        if self.current_filename == "yelp":
                self.display_final_labeled_data()
        elif self.current_filename == "ag_news":
                self.display_ag_news_labeled_data()
        elif self.current_filename=="imdb" :
                self.display_imdb_labeled_data()
        self.save_button.configure(state="normal", fg_color="#1f6aa5")
        self.text_display.see("end")




#Yelp p/n  (new label star csv )nin summary statistics e gömüldüğü part
    def display_final_labeled_data(self):
        try:
            df = pd.read_csv("yelp_400_labeled_final1.csv")
            self.current_df = df #csv olarak  kaydetmek için eklendi
            self.current_filename = "yelp_summary"
            self.save_button.configure(state="normal", fg_color="#1f6aa5")
            display_df = df[['new_label_star', 'sentiment', 'text']].copy()
            display_df['text'] = display_df['text'].astype(str).str.slice(0, 55) + "..."
            table = tabulate(display_df.head(400), headers='keys', tablefmt='psql', showindex=True)

            # Başına bir satır boşluk ekledim ki istatistiklerle yapışmasın
            self.text_display.insert("end", "\n--- FINAL LABELED RESEARCH DATASET (YELP) ---\n\n" + table + "\n")
            self.status_label.configure(text="Status: Research data loaded! ✨", text_color="#00FF00")
            self.save_button.configure(state="normal", fg_color="#1f6aa5")
        except:
            self.status_label.configure(text="Status: Labeled file not found!", text_color="red")
#Ag news p/n ag news labeled final csv nin summary statistics butonuna gömüldüğü part
    def display_ag_news_labeled_data(self):

        try:
            # 1. Dosyayı güvenli şekilde okur
            path = "ag_news_383_labeled_final.csv"
            if not os.path.exists(path):
                path = os.path.join("indirilen_veriler", path)

            df = pd.read_csv(path)
            self.current_df = df # csv olarak kaydetmek için eklendi
            self.current_filename = "ag_news_summary"
            self.save_button.configure(state="normal", fg_color="#1f6aa5")

            # 2. Metin Sütununu Tespit Et (Kısa ve Temiz Yol)
            potential = ['description', 'text', 'Title']
            # Listede olanı seç, yoksa sistem sütunları dışındaki ilk sütunu al
            text_col = next((c for c in potential if c in df.columns),
                            [c for c in df.columns if c not in ['label', 'sentiment', 'Unnamed: 0']][0])

            # 3. Tabloyu oluşturur ve kırpar
            display_df = df[['label', 'sentiment', text_col]].copy()
            display_df[text_col] = display_df[text_col].astype(str).str.slice(0, 80) + "..."

            table = tabulate(display_df.head(383), headers='keys', tablefmt='psql', showindex=True)

            # 4. Arayüze Yazdırır
            self.text_display.insert("end", f"\n--- AG NEWS RESEARCH DATASET ---\n\n{table}\n")
            self.status_label.configure(text="Status: Loaded! 🚀", text_color="#00FF00")
            self.save_button.configure(state="normal")

        except Exception as e:
            self.status_label.configure(text="Display Error!", text_color="red")
            print(f"Hata: {e}")

    def display_imdb_labeled_data(self):
        try:
            df = pd.read_csv("imdb_385_labeled_final.csv")
            self.current_df = df
            self.current_filename = "imdb_summary"
            self.save_button.configure(state="normal", fg_color="#1f6aa5")
            display_df = df[['label', 'sentiment', 'text']].copy()
            display_df['text'] = display_df['text'].astype(str).str.slice(0, 80) + "..."
            table = tabulate(display_df.head(385), headers='keys', tablefmt='psql', showindex=True)
            self.text_display.insert("end", f"\n--- IMDB RESEARCH DATASET ---\n\n{table}\n")
            self.status_label.configure(text="Status: IMDB data loaded! 🎬", text_color="#00FF00")
            self.save_button.configure(state="normal")
        except Exception as e:
            self.status_label.configure(text="Display Error!", text_color="red")
            print(f"Hata: {e}")

# Yelp Llm  prediction part
    def run_llm_prediction(self):
        labeled_file = "yelp_400_labeled_final1.csv"
        test_file = "yelp_standard_sample_400.csv"

        if not os.path.exists(labeled_file):
            messagebox.showerror("Error", "Please organize the data using Summary Statistics first!")
            return

        try:
            self.status_label.configure(text="Status: Learning from your labels...", text_color="orange")
            self.update()

            # 1. Veriyi Oku
            full_labeled_df = pd.read_csv(labeled_file)

            # --- EKLEME: Gerçekçi etkinlik ölçümü için veriyi bölüyoruz ---
            # Verinin %80'i ile öğrenecek, %20'si ile kendini test edecek
            train_df, val_df = train_test_split(full_labeled_df, test_size=0.2, random_state=42)

            model = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', LogisticRegression())
            ])

            # 2. Modeli Eğit (Sadece eğitim kısmıyla)
            model.fit(train_df['text'].astype(str), train_df['sentiment'])

            # 3. Tahmin Yap (Standart veri üzerinde)
            test_df = pd.read_csv(test_file)
            predictions = model.predict(test_df['text'].astype(str))

            test_df['predicted_sentiment'] = predictions
            self.current_df = test_df
            self.current_filename = "yelp_predictions"

            self.text_display.delete("1.0", "end")
            self.text_display.insert("end", "📊 --- PREDICTION BASED ON YOUR ORGANIZED DATA --- 📊\n")

            # --- EKLEME: Etkinlik Raporunu Arayüze Yazdır ---
            from sklearn.metrics import classification_report
            val_preds = model.predict(val_df['text'].astype(str))
            report = classification_report(val_df['sentiment'], val_preds)
            self.text_display.insert("end", f"Model Efficiency (Validation Results):\n{report}\n")
            self.text_display.insert("end", "=" * 65 + "\n\n")

            self.text_display.insert("end", f"Training Source: {labeled_file}\n")
            self.text_display.insert("end", "Method: Supervised Learning (Trained by Nuray's Labels)\n")
            self.text_display.insert("end", "=" * 65 + "\n\n")

            for i in range((len(test_df))):
                text = str(test_df['text'].iloc[i])
                pred = predictions[i]
                decision = "Positive (p)" if pred == 'p' else "Negative (n)"
                preview = text[:75].replace('\n', ' ').strip()
                self.text_display.insert("end", f"[{i + 1}] {preview}...\n")
                self.text_display.insert("end", f"    ▶ YOUR MODEL'S DECISION: {decision}\n")
                self.text_display.insert("end", "-" * 65 + "\n")

            self.status_label.configure(text="Status: Prediction complete! ✨", text_color="#00FF00")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

        self.save_button.configure(state="normal", fg_color="#1f6aa5")
  #Ag news Llm prediction part
    def run_ag_news_prediction(self):
        labeled_file = "ag_news_383_labeled_final.csv"
        test_file = "ag_news_standard_sample_383.csv"

        if not os.path.exists(labeled_file):
            messagebox.showerror("Error", "Please organize the data using Summary Statistics first!")
            return

        try:
            self.status_label.configure(text="Status: Learning & Testing...", text_color="orange")
            self.update()

            # 1. Veriyi Oku ve Karıştır
            full_labeled_df = pd.read_csv(labeled_file)
            text_col = 'description' if 'description' in full_labeled_df.columns else 'text'

            #  Veriyi %80 Eğitim, %20 Test olarak bölüyoruz burda
            train_df, val_df = train_test_split(full_labeled_df, test_size=0.2, random_state=42)

            model = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', LogisticRegression())
            ])

            # 2. Modeli Sadece Eğitim Verisiyle Eğitiyoruz
            model.fit(train_df[text_col].astype(str), train_df['sentiment'])

            # 3. Standart Örnek Üzerinde Tahmin Yaptırıyoruz
            test_df = pd.read_csv(test_file)
            predictions = model.predict(test_df[text_col].astype(str))

            test_df['predicted_sentiment'] = predictions
            self.current_df = test_df
            self.current_filename = "ag_news_predictions"

            self.text_display.delete("1.0", "end")
            self.text_display.insert("end", "📊 --- PREDICTION BASED ON YOUR ORGANIZED DATA --- 📊\n")

            #  Gerçek Etkinlik Raporunu Yazdııyoruz
            from sklearn.metrics import classification_report
            val_preds = model.predict(val_df[text_col].astype(str))
            report = classification_report(val_df['sentiment'], val_preds)
            self.text_display.insert("end", f"Model Efficiency (Validation Results):\n{report}\n")
            self.text_display.insert("end", "=" * 65 + "\n\n")

            self.text_display.insert("end", f"Training Source: {labeled_file}\n")
            self.text_display.insert("end", "Method: Supervised Learning (Trained by Nuray's Labels)\n")
            self.text_display.insert("end", "=" * 65 + "\n\n")

            for i in range(len(test_df)):
                text = str(test_df[text_col].iloc[i])
                pred = predictions[i]
                decision = "Positive (p)" if pred == 'p' else "Negative (n)"
                preview = text[:75].replace('\n', ' ').strip()
                self.text_display.insert("end", f"[{i + 1}] {preview}...\n")
                self.text_display.insert("end", f"    ▶ YOUR MODEL'S DECISION: {decision}\n")
                self.text_display.insert("end", "-" * 65 + "\n")

            self.status_label.configure(text="Status: Prediction complete! ✨", text_color="#00FF00")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

        self.save_button.configure(state="normal", fg_color="#1f6aa5")
#imdb llm prediction part
    def run_imdb_prediction(self):
        labeled_file = "imdb_385_labeled_final.csv"
        test_file = "imdb_standard_sample_385.csv"

        if not os.path.exists(labeled_file):
            messagebox.showerror("Error", "Please organize the data using Summary Statistics first!")
            return

        try:
            self.status_label.configure(text="Status: Learning & Testing...", text_color="orange")
            self.update()

            # 1. Veriyi Okutuyoruz
            full_labeled_df = pd.read_csv(labeled_file)

            #  Veriyi %80 Eğitim, %20 Test olarak bölüyoruz
            # Bu işlem "gri" duran train_test_split kütüphanesini aktif eder.
            train_df, val_df = train_test_split(full_labeled_df, test_size=0.2, random_state=42)

            model = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', LogisticRegression())
            ])

            # 2. Modeli Sadece Eğitim Verisiyle Eğitiyoruz
            model.fit(train_df['text'].astype(str), train_df['sentiment'])

            # 3. Standart Örnek Üzerinde Tahmin Yaptırıyoruz
            test_df = pd.read_csv(test_file)
            predictions = model.predict(test_df['text'].astype(str))

            test_df['predicted_sentiment'] = predictions
            self.current_df = test_df
            self.current_filename = "imdb_predictions"

            self.text_display.delete("1.0", "end")
            self.text_display.insert("end", "📊 --- PREDICTION BASED ON YOUR ORGANIZED DATA --- 📊\n")

            #  Gerçek Etkinlik Raporunu (Validation) Yazdırtıyoruz
            from sklearn.metrics import classification_report
            val_preds = model.predict(val_df['text'].astype(str))
            report = classification_report(val_df['sentiment'], val_preds)
            self.text_display.insert("end", f"Model Efficiency (Validation Results):\n{report}\n")
            self.text_display.insert("end", "=" * 65 + "\n\n")

            self.text_display.insert("end", f"Training Source: {labeled_file}\n")
            self.text_display.insert("end", "Method: Supervised Learning (Trained by Nuray's Labels)\n")
            self.text_display.insert("end", "=" * 65 + "\n\n")

            for i in range(len(test_df)):
                text = str(test_df['text'].iloc[i])
                pred = predictions[i]
                decision = "Positive (p)" if pred == 'p' else "Negative (n)"
                preview = text[:75].replace('\n', ' ').strip()
                self.text_display.insert("end", f"[{i + 1}] {preview}...\n")
                self.text_display.insert("end", f"    ▶ YOUR MODEL'S DECISION: {decision}\n")
                self.text_display.insert("end", "-" * 65 + "\n")

            self.status_label.configure(text="Status: Prediction complete! ✨", text_color="#00FF00")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

        self.save_button.configure(state="normal", fg_color="#1f6aa5")
if __name__ == "__main__":
    app = DataExplor()
    #app.display_final_labeled_data()
    app.mainloop()


# Sentiment Analysis

# def create_standard_sample():
# df_sample, _ = sample_yelp()
# df_sample.to_csv("yelp_standard_sample_400.csv", index=True, encoding="utf-8-sig")

# create_standard_sample()

def apply_sentiment_and_save():
    df = pd.read_csv("yelp_standard_sample_400.csv", index_col=0)
    df['sentiment'] = ""
    df.loc[0, 'sentiment'] = 'n'
    df.loc[1, 'sentiment'] = 'n'
    df.loc[2, 'sentiment'] = 'n'
    df.loc[3, 'sentiment'] = 'n'
    df.loc[4, 'sentiment'] = 'n'
    df.loc[5, 'sentiment'] = 'n'
    df.loc[6, 'sentiment'] = 'n'
    df.loc[7, 'sentiment'] = 'n'
    df.loc[8, 'sentiment'] = 'n'
    df.loc[9, 'sentiment'] = 'n'
    df.loc[10, 'sentiment'] = 'n'
    df.loc[11, 'sentiment'] = 'n'
    df.loc[12, 'sentiment'] = 'n'
    df.loc[13, 'sentiment'] = 'n'
    df.loc[14, 'sentiment'] = 'n'
    df.loc[15, 'sentiment'] = 'n'
    df.loc[16, 'sentiment'] = 'n'
    df.loc[17, 'sentiment'] = 'n'
    df.loc[18, 'sentiment'] = 'n'
    df.loc[19, 'sentiment'] = 'n'
    df.loc[20, 'sentiment'] = 'n'
    df.loc[21, 'sentiment'] = 'n'
    df.loc[22, 'sentiment'] = 'n'
    df.loc[23, 'sentiment'] = 'n'
    df.loc[24, 'sentiment'] = 'n'
    df.loc[25, 'sentiment'] = 'n'
    df.loc[26, 'sentiment'] = 'n'
    df.loc[27, 'sentiment'] = 'n'
    df.loc[28, 'sentiment'] = 'n'
    df.loc[29, 'sentiment'] = 'n'
    df.loc[30, 'sentiment'] = 'n'
    df.loc[31, 'sentiment'] = 'n'
    df.loc[32, 'sentiment'] = 'n'
    df.loc[33, 'sentiment'] = 'n'
    df.loc[34, 'sentiment'] = 'n'
    df.loc[35, 'sentiment'] = 'n'
    df.loc[36, 'sentiment'] = 'n'
    df.loc[37, 'sentiment'] = 'n'
    df.loc[38, 'sentiment'] = 'n'
    df.loc[39, 'sentiment'] = 'n'
    df.loc[40, 'sentiment'] = 'n'
    df.loc[41, 'sentiment'] = 'n'
    df.loc[42, 'sentiment'] = 'n'
    df.loc[43, 'sentiment'] = 'n'
    df.loc[44, 'sentiment'] = 'n'
    df.loc[45, 'sentiment'] = 'n'
    df.loc[46, 'sentiment'] = 'n'
    df.loc[47, 'sentiment'] = 'n'
    df.loc[48, 'sentiment'] = 'n'
    df.loc[49, 'sentiment'] = 'n'
    df.loc[50, 'sentiment'] = 'n'
    df.loc[51, 'sentiment'] = 'n'
    df.loc[52, 'sentiment'] = 'n'
    df.loc[53, 'sentiment'] = 'n'
    df.loc[54, 'sentiment'] = 'n'
    df.loc[55, 'sentiment'] = 'n'
    df.loc[56, 'sentiment'] = 'n'
    df.loc[57, 'sentiment'] = 'n'
    df.loc[58, 'sentiment'] = 'n'
    df.loc[59, 'sentiment'] = 'n'
    df.loc[60, 'sentiment'] = 'n'
    df.loc[61, 'sentiment'] = 'n'
    df.loc[62, 'sentiment'] = 'n'
    df.loc[63, 'sentiment'] = 'n'
    df.loc[64, 'sentiment'] = 'n'
    df.loc[65, 'sentiment'] = 'n'
    df.loc[66, 'sentiment'] = 'n'
    df.loc[67, 'sentiment'] = 'n'
    df.loc[68, 'sentiment'] = 'n'
    df.loc[69, 'sentiment'] = 'n'
    df.loc[70, 'sentiment'] = 'n'
    df.loc[71, 'sentiment'] = 'n'
    df.loc[72, 'sentiment'] = 'n'
    df.loc[73, 'sentiment'] = 'n'
    df.loc[74, 'sentiment'] = 'n'
    df.loc[75, 'sentiment'] = 'n'
    df.loc[76, 'sentiment'] = 'n'
    df.loc[77, 'sentiment'] = 'n'
    df.loc[78, 'sentiment'] = 'n'
    df.loc[79, 'sentiment'] = 'n'
    df.loc[80, 'sentiment'] = 'n'
    df.loc[81, 'sentiment'] = 'n'
    df.loc[82, 'sentiment'] = 'n'
    df.loc[83, 'sentiment'] = 'n'
    df.loc[84, 'sentiment'] = 'n'
    df.loc[85, 'sentiment'] = 'n'
    df.loc[86, 'sentiment'] = 'n'
    df.loc[87, 'sentiment'] = 'n'
    df.loc[88, 'sentiment'] = 'n'
    df.loc[89, 'sentiment'] = 'n'
    df.loc[90, 'sentiment'] = 'n'
    df.loc[91, 'sentiment'] = 'n'
    df.loc[92, 'sentiment'] = 'n'
    df.loc[93, 'sentiment'] = 'n'
    df.loc[94, 'sentiment'] = 'n'
    df.loc[95, 'sentiment'] = 'n'
    df.loc[96, 'sentiment'] = 'n'
    df.loc[97, 'sentiment'] = 'n'
    df.loc[98, 'sentiment'] = 'n'
    df.loc[99, 'sentiment'] = 'n'
    df.loc[100, 'sentiment'] = 'n'
    df.loc[101, 'sentiment'] = 'n'
    df.loc[102, 'sentiment'] = 'n'
    df.loc[103, 'sentiment'] = 'n'
    df.loc[104, 'sentiment'] = 'n'
    df.loc[105, 'sentiment'] = 'n'
    df.loc[106, 'sentiment'] = 'n'
    df.loc[107, 'sentiment'] = 'n'
    df.loc[108, 'sentiment'] = 'n'
    df.loc[109, 'sentiment'] = 'n'
    df.loc[110, 'sentiment'] = 'n'
    df.loc[111, 'sentiment'] = 'n'
    df.loc[112, 'sentiment'] = 'n'
    df.loc[113, 'sentiment'] = 'n'
    df.loc[114, 'sentiment'] = 'n'
    df.loc[115, 'sentiment'] = 'n'
    df.loc[116, 'sentiment'] = 'n'
    df.loc[117, 'sentiment'] = 'n'
    df.loc[118, 'sentiment'] = 'n'
    df.loc[119, 'sentiment'] = 'n'
    df.loc[120, 'sentiment'] = 'n'
    df.loc[121, 'sentiment'] = 'n'
    df.loc[122, 'sentiment'] = 'n'
    df.loc[123, 'sentiment'] = 'n'
    df.loc[124, 'sentiment'] = 'n'
    df.loc[125, 'sentiment'] = 'n'
    df.loc[126, 'sentiment'] = 'n'
    df.loc[127, 'sentiment'] = 'n'
    df.loc[128, 'sentiment'] = 'n'
    df.loc[129, 'sentiment'] = 'n'
    df.loc[130, 'sentiment'] = 'n'
    df.loc[131, 'sentiment'] = 'n'
    df.loc[132, 'sentiment'] = 'n'
    df.loc[133, 'sentiment'] = 'n'
    df.loc[134, 'sentiment'] = 'n'
    df.loc[135, 'sentiment'] = 'n'
    df.loc[136, 'sentiment'] = 'n'
    df.loc[137, 'sentiment'] = 'n'
    df.loc[138, 'sentiment'] = 'n'
    df.loc[139, 'sentiment'] = 'n'
    df.loc[140, 'sentiment'] = 'n'
    df.loc[141, 'sentiment'] = 'n'
    df.loc[142, 'sentiment'] = 'n'
    df.loc[143, 'sentiment'] = 'n'
    df.loc[144, 'sentiment'] = 'n'
    df.loc[145, 'sentiment'] = 'n'
    df.loc[146, 'sentiment'] = 'n'
    df.loc[147, 'sentiment'] = 'n'
    df.loc[148, 'sentiment'] = 'n'
    df.loc[149, 'sentiment'] = 'n'
    df.loc[150, 'sentiment'] = 'n'
    df.loc[151, 'sentiment'] = 'n'
    df.loc[152, 'sentiment'] = 'n'
    df.loc[153, 'sentiment'] = 'n'
    df.loc[154, 'sentiment'] = 'n'
    df.loc[155, 'sentiment'] = 'n'
    df.loc[156, 'sentiment'] = 'n'
    df.loc[157, 'sentiment'] = 'n'
    df.loc[158, 'sentiment'] = 'n'
    df.loc[159, 'sentiment'] = 'n'
    df.loc[160, 'sentiment'] = 'n'
    df.loc[161, 'sentiment'] = 'n'
    df.loc[162, 'sentiment'] = 'n'
    df.loc[163, 'sentiment'] = 'n'
    df.loc[164, 'sentiment'] = 'n'
    df.loc[165, 'sentiment'] = 'n'
    df.loc[166, 'sentiment'] = 'n'
    df.loc[167, 'sentiment'] = 'n'
    df.loc[168, 'sentiment'] = 'n'
    df.loc[169, 'sentiment'] = 'n'
    df.loc[170, 'sentiment'] = 'n'
    df.loc[171, 'sentiment'] = 'n'
    df.loc[172, 'sentiment'] = 'n'
    df.loc[173, 'sentiment'] = 'n'
    df.loc[174, 'sentiment'] = 'p'  # n
    df.loc[175, 'sentiment'] = 'p'
    df.loc[176, 'sentiment'] = 'p'
    df.loc[177, 'sentiment'] = 'p'
    df.loc[178, 'sentiment'] = 'p'
    df.loc[179, 'sentiment'] = 'p'
    df.loc[180, 'sentiment'] = 'p'
    df.loc[181, 'sentiment'] = 'p'
    df.loc[182, 'sentiment'] = 'p'
    df.loc[183, 'sentiment'] = 'p'
    df.loc[184, 'sentiment'] = 'p'
    df.loc[185, 'sentiment'] = 'p'
    df.loc[186, 'sentiment'] = 'p'
    df.loc[187, 'sentiment'] = 'p'
    df.loc[188, 'sentiment'] = 'p'
    df.loc[189, 'sentiment'] = 'p'
    df.loc[190, 'sentiment'] = 'p'
    df.loc[191, 'sentiment'] = 'p'
    df.loc[192, 'sentiment'] = 'p'
    df.loc[193, 'sentiment'] = 'p'
    df.loc[194, 'sentiment'] = 'p'
    df.loc[195, 'sentiment'] = 'p'
    df.loc[196, 'sentiment'] = 'p'
    df.loc[197, 'sentiment'] = 'p'
    df.loc[198, 'sentiment'] = 'p'
    df.loc[199, 'sentiment'] = 'p'
    df.loc[200, 'sentiment'] = 'p'
    df.loc[201, 'sentiment'] = 'p'
    df.loc[202, 'sentiment'] = 'p'
    df.loc[203, 'sentiment'] = 'p'
    df.loc[204, 'sentiment'] = 'p'
    df.loc[205, 'sentiment'] = 'p'
    df.loc[206, 'sentiment'] = 'p'
    df.loc[207, 'sentiment'] = 'p'
    df.loc[208, 'sentiment'] = 'p'
    df.loc[209, 'sentiment'] = 'p'
    df.loc[210, 'sentiment'] = 'p'
    df.loc[211, 'sentiment'] = 'p'
    df.loc[212, 'sentiment'] = 'p'
    df.loc[213, 'sentiment'] = 'p'
    df.loc[214, 'sentiment'] = 'p'
    df.loc[215, 'sentiment'] = 'p'
    df.loc[216, 'sentiment'] = 'p'
    df.loc[217, 'sentiment'] = 'p'
    df.loc[218, 'sentiment'] = 'p'
    df.loc[219, 'sentiment'] = 'p'
    df.loc[220, 'sentiment'] = 'p'
    df.loc[221, 'sentiment'] = 'p'
    df.loc[222, 'sentiment'] = 'p'
    df.loc[223, 'sentiment'] = 'p'
    df.loc[224, 'sentiment'] = 'p'
    df.loc[225, 'sentiment'] = 'p'
    df.loc[226, 'sentiment'] = 'p'
    df.loc[227, 'sentiment'] = 'p'
    df.loc[228, 'sentiment'] = 'p'
    df.loc[229, 'sentiment'] = 'p'
    df.loc[230, 'sentiment'] = 'p'
    df.loc[231, 'sentiment'] = 'p'
    df.loc[232, 'sentiment'] = 'p'
    df.loc[233, 'sentiment'] = 'p'
    df.loc[234, 'sentiment'] = 'p'
    df.loc[235, 'sentiment'] = 'p'
    df.loc[236, 'sentiment'] = 'p'
    df.loc[237, 'sentiment'] = 'p'
    df.loc[238, 'sentiment'] = 'p'
    df.loc[239, 'sentiment'] = 'p'
    df.loc[240, 'sentiment'] = 'p'
    df.loc[241, 'sentiment'] = 'p'
    df.loc[242, 'sentiment'] = 'p'
    df.loc[243, 'sentiment'] = 'p'
    df.loc[244, 'sentiment'] = 'p'
    df.loc[245, 'sentiment'] = 'p'
    df.loc[246, 'sentiment'] = 'p'
    df.loc[247, 'sentiment'] = 'p'
    df.loc[248, 'sentiment'] = 'p'
    df.loc[249, 'sentiment'] = 'p'
    df.loc[250, 'sentiment'] = 'p'
    df.loc[251, 'sentiment'] = 'p'  # n
    df.loc[252, 'sentiment'] = 'p'
    df.loc[253, 'sentiment'] = 'p'
    df.loc[254, 'sentiment'] = 'p'
    df.loc[255, 'sentiment'] = 'p'
    df.loc[256, 'sentiment'] = 'p'
    df.loc[257, 'sentiment'] = 'p'
    df.loc[258, 'sentiment'] = 'p'
    df.loc[259, 'sentiment'] = 'p'
    df.loc[260, 'sentiment'] = 'p'
    df.loc[261, 'sentiment'] = 'p'
    df.loc[262, 'sentiment'] = 'p'
    df.loc[263, 'sentiment'] = 'p'
    df.loc[264, 'sentiment'] = 'p'
    df.loc[265, 'sentiment'] = 'p'
    df.loc[266, 'sentiment'] = 'p'
    df.loc[267, 'sentiment'] = 'p'
    df.loc[268, 'sentiment'] = 'p'
    df.loc[269, 'sentiment'] = 'p'
    df.loc[270, 'sentiment'] = 'p'
    df.loc[271, 'sentiment'] = 'p'
    df.loc[272, 'sentiment'] = 'p'
    df.loc[273, 'sentiment'] = 'p'
    df.loc[274, 'sentiment'] = 'p'
    df.loc[275, 'sentiment'] = 'p'
    df.loc[276, 'sentiment'] = 'p'
    df.loc[277, 'sentiment'] = 'p'
    df.loc[278, 'sentiment'] = 'p'
    df.loc[279, 'sentiment'] = 'p'
    df.loc[280, 'sentiment'] = 'p'
    df.loc[281, 'sentiment'] = 'p'
    df.loc[282, 'sentiment'] = 'p'
    df.loc[283, 'sentiment'] = 'p'
    df.loc[284, 'sentiment'] = 'p'
    df.loc[285, 'sentiment'] = 'p'
    df.loc[286, 'sentiment'] = 'p'
    df.loc[287, 'sentiment'] = 'p'
    df.loc[288, 'sentiment'] = 'p'
    df.loc[289, 'sentiment'] = 'p'
    df.loc[290, 'sentiment'] = 'p'
    df.loc[291, 'sentiment'] = 'p'
    df.loc[292, 'sentiment'] = 'p'
    df.loc[293, 'sentiment'] = 'p'
    df.loc[294, 'sentiment'] = 'p'
    df.loc[295, 'sentiment'] = 'p'
    df.loc[296, 'sentiment'] = 'p'
    df.loc[297, 'sentiment'] = 'p'
    df.loc[298, 'sentiment'] = 'p'
    df.loc[299, 'sentiment'] = 'p'
    df.loc[300, 'sentiment'] = 'p'
    df.loc[301, 'sentiment'] = 'p'
    df.loc[302, 'sentiment'] = 'p'
    df.loc[303, 'sentiment'] = 'p'
    df.loc[304, 'sentiment'] = 'p'
    df.loc[305, 'sentiment'] = 'p'
    df.loc[306, 'sentiment'] = 'p'
    df.loc[307, 'sentiment'] = 'p'
    df.loc[308, 'sentiment'] = 'p'
    df.loc[309, 'sentiment'] = 'p'
    df.loc[310, 'sentiment'] = 'p'
    df.loc[311, 'sentiment'] = 'p'
    df.loc[312, 'sentiment'] = 'p'
    df.loc[313, 'sentiment'] = 'p'
    df.loc[314, 'sentiment'] = 'p'
    df.loc[315, 'sentiment'] = 'p'
    df.loc[316, 'sentiment'] = 'p'
    df.loc[317, 'sentiment'] = 'p'
    df.loc[318, 'sentiment'] = 'p'
    df.loc[319, 'sentiment'] = 'p'
    df.loc[320, 'sentiment'] = 'p'
    df.loc[321, 'sentiment'] = 'p'
    df.loc[322, 'sentiment'] = 'p'
    df.loc[323, 'sentiment'] = 'p'
    df.loc[324, 'sentiment'] = 'p'
    df.loc[325, 'sentiment'] = 'p'
    df.loc[326, 'sentiment'] = 'p'
    df.loc[327, 'sentiment'] = 'p'
    df.loc[328, 'sentiment'] = 'p'
    df.loc[329, 'sentiment'] = 'p'
    df.loc[330, 'sentiment'] = 'p'
    df.loc[331, 'sentiment'] = 'p'
    df.loc[332, 'sentiment'] = 'p'
    df.loc[333, 'sentiment'] = 'p'
    df.loc[334, 'sentiment'] = 'p'
    df.loc[335, 'sentiment'] = 'p'
    df.loc[336, 'sentiment'] = 'p'
    df.loc[337, 'sentiment'] = 'p'
    df.loc[338, 'sentiment'] = 'p'
    df.loc[339, 'sentiment'] = 'p'
    df.loc[340, 'sentiment'] = 'p'
    df.loc[341, 'sentiment'] = 'p'
    df.loc[342, 'sentiment'] = 'p'
    df.loc[343, 'sentiment'] = 'p'
    df.loc[344, 'sentiment'] = 'p'
    df.loc[345, 'sentiment'] = 'p'
    df.loc[346, 'sentiment'] = 'p'
    df.loc[347, 'sentiment'] = 'p'
    df.loc[348, 'sentiment'] = 'p'
    df.loc[349, 'sentiment'] = 'p'
    df.loc[350, 'sentiment'] = 'p'
    df.loc[351, 'sentiment'] = 'p'
    df.loc[352, 'sentiment'] = 'p'
    df.loc[353, 'sentiment'] = 'p'
    df.loc[354, 'sentiment'] = 'p'
    df.loc[355, 'sentiment'] = 'p'
    df.loc[356, 'sentiment'] = 'p'
    df.loc[357, 'sentiment'] = 'p'
    df.loc[358, 'sentiment'] = 'p'
    df.loc[359, 'sentiment'] = 'p'
    df.loc[360, 'sentiment'] = 'p'
    df.loc[361, 'sentiment'] = 'p'
    df.loc[362, 'sentiment'] = 'p'
    df.loc[363, 'sentiment'] = 'p'
    df.loc[364, 'sentiment'] = 'p'
    df.loc[365, 'sentiment'] = 'p'
    df.loc[366, 'sentiment'] = 'p'
    df.loc[367, 'sentiment'] = 'p'
    df.loc[368, 'sentiment'] = 'p'
    df.loc[369, 'sentiment'] = 'p'
    df.loc[370, 'sentiment'] = 'p'
    df.loc[371, 'sentiment'] = 'p'
    df.loc[372, 'sentiment'] = 'p'
    df.loc[373, 'sentiment'] = 'p'
    df.loc[374, 'sentiment'] = 'p'
    df.loc[375, 'sentiment'] = 'p'
    df.loc[376, 'sentiment'] = 'p'
    df.loc[377, 'sentiment'] = 'p'
    df.loc[378, 'sentiment'] = 'p'
    df.loc[379, 'sentiment'] = 'p'
    df.loc[380, 'sentiment'] = 'p'
    df.loc[381, 'sentiment'] = 'p'
    df.loc[382, 'sentiment'] = 'p'
    df.loc[383, 'sentiment'] = 'p'
    df.loc[384, 'sentiment'] = 'p'
    df.loc[385, 'sentiment'] = 'p'
    df.loc[386, 'sentiment'] = 'p'
    df.loc[387, 'sentiment'] = 'p'
    df.loc[388, 'sentiment'] = 'p'
    df.loc[389, 'sentiment'] = 'p'
    df.loc[390, 'sentiment'] = 'p'
    df.loc[391, 'sentiment'] = 'p'
    df.loc[392, 'sentiment'] = 'p'
    df.loc[393, 'sentiment'] = 'p'
    df.loc[394, 'sentiment'] = 'p'
    df.loc[395, 'sentiment'] = 'p'
    df.loc[396, 'sentiment'] = 'p'
    df.loc[397, 'sentiment'] = 'p'
    df.loc[398, 'sentiment'] = 'p'
    df.loc[399, 'sentiment'] = 'p'

    df.to_csv("yelp_400_labeled_final1.csv", index=False, encoding="utf-8-sig")

#apply_sentiment_and_save()

# LLM Ag_News
#def create_ag_news_standard_sample():
     #df_sample, _ = sample_ag_news()

     #df_sample.head(383).to_csv("ag_news_standard_sample_383.csv", index=True, encoding="utf-8-sig")

#create_ag_news_standard_sample()
def apply_sentiment_and_save():
    df = pd.read_csv("ag_news_standard_sample_383.csv", index_col=0)
    df['sentiment'] = ""
    df.loc[0, 'sentiment']=  'n'
    df.loc[1, 'sentiment']=  'n'
    df.loc[2, 'sentiment']=  'p' #n
    df.loc[3, 'sentiment']=  'p' #n
    df.loc[4, 'sentiment']=  'p'
    df.loc[5, 'sentiment']=  'p' #n
    df.loc[6, 'sentiment']=  'p'
    df.loc[7, 'sentiment']=  'n'
    df.loc[8, 'sentiment']=  'n'
    df.loc[9, 'sentiment']=  'n'
    df.loc[10, 'sentiment']= 'n'
    df.loc[11, 'sentiment']= 'p'
    df.loc[12, 'sentiment']= 'n'#p
    df.loc[13, 'sentiment']= 'n'
    df.loc[14, 'sentiment']= 'p' #n
    df.loc[15, 'sentiment']= 'p'
    df.loc[16, 'sentiment']= 'p'
    df.loc[17, 'sentiment']= 'p'
    df.loc[18, 'sentiment']= 'p'
    df.loc[19, 'sentiment']= 'p'
    df.loc[20, 'sentiment']= 'p'
    df.loc[21, 'sentiment'] = 'n'
    df.loc[22, 'sentiment'] = 'p'
    df.loc[23, 'sentiment'] = 'p'
    df.loc[24, 'sentiment'] = 'n'
    df.loc[25, 'sentiment'] = 'n'
    df.loc[26, 'sentiment'] = 'p'
    df.loc[27, 'sentiment'] = 'n'
    df.loc[28, 'sentiment'] = 'n'
    df.loc[29, 'sentiment'] = 'n'
    df.loc[30, 'sentiment'] = 'p'
    df.loc[31, 'sentiment'] = 'p'
    df.loc[32, 'sentiment'] = 'p'
    df.loc[33, 'sentiment'] = 'p'
    df.loc[34, 'sentiment'] = 'n'
    df.loc[35, 'sentiment'] = 'p'
    df.loc[36, 'sentiment'] = 'p'
    df.loc[37, 'sentiment'] = 'p'
    df.loc[38, 'sentiment'] = 'n'
    df.loc[39, 'sentiment'] = 'n'
    df.loc[40, 'sentiment'] = 'p'
    df.loc[41, 'sentiment'] = 'p'
    df.loc[42, 'sentiment'] = 'n'
    df.loc[43, 'sentiment'] = 'p'
    df.loc[44, 'sentiment'] = 'p'
    df.loc[45, 'sentiment'] = 'p'
    df.loc[46, 'sentiment'] = 'n'
    df.loc[47, 'sentiment'] = 'n'
    df.loc[48, 'sentiment'] = 'p'
    df.loc[49, 'sentiment'] = 'p'
    df.loc[50, 'sentiment'] = 'p'
    df.loc[51, 'sentiment'] = 'p' #n
    df.loc[52, 'sentiment'] = 'n'
    df.loc[53, 'sentiment'] = 'p'
    df.loc[54, 'sentiment'] = 'n'
    df.loc[55, 'sentiment'] = 'p'
    df.loc[56, 'sentiment'] = 'n'
    df.loc[57, 'sentiment'] = 'p'
    df.loc[58, 'sentiment'] = 'p'
    df.loc[59, 'sentiment'] = 'p'
    df.loc[60, 'sentiment'] = 'p' #n
    df.loc[61, 'sentiment'] = 'p'
    df.loc[62, 'sentiment'] = 'n'
    df.loc[63, 'sentiment'] = 'n'
    df.loc[64, 'sentiment'] = 'p'
    df.loc[65, 'sentiment'] = 'n'
    df.loc[66, 'sentiment'] = 'p'
    df.loc[67, 'sentiment'] = 'p'
    df.loc[68, 'sentiment'] = 'n'
    df.loc[69, 'sentiment'] = 'n'
    df.loc[70, 'sentiment'] = 'n'
    df.loc[71, 'sentiment'] = 'p'
    df.loc[72, 'sentiment'] = 'n'
    df.loc[73, 'sentiment'] = 'p'
    df.loc[74, 'sentiment'] = 'p'
    df.loc[75, 'sentiment'] = 'p'
    df.loc[76, 'sentiment'] = 'p'
    df.loc[77, 'sentiment'] = 'p'
    df.loc[78, 'sentiment'] = 'p'
    df.loc[79, 'sentiment'] = 'p'
    df.loc[80, 'sentiment'] = 'n'
    df.loc[81, 'sentiment'] = 'n'
    df.loc[82, 'sentiment'] = 'p'
    df.loc[83, 'sentiment'] = 'p'
    df.loc[84, 'sentiment'] = 'p'
    df.loc[85, 'sentiment'] = 'n'
    df.loc[86, 'sentiment'] = 'p'
    df.loc[87, 'sentiment'] = 'p'
    df.loc[88, 'sentiment'] = 'n'
    df.loc[89, 'sentiment'] = 'n'
    df.loc[90, 'sentiment'] = 'n'
    df.loc[91, 'sentiment'] = 'n'
    df.loc[92, 'sentiment'] = 'n'
    df.loc[93, 'sentiment'] = 'p'
    df.loc[94, 'sentiment'] = 'p'
    df.loc[95, 'sentiment'] = 'p'
    df.loc[96, 'sentiment'] = 'n'
    df.loc[97, 'sentiment'] = 'p'
    df.loc[98, 'sentiment'] = 'p'
    df.loc[99, 'sentiment'] = 'p'
    df.loc[100, 'sentiment'] = 'n'
    df.loc[101, 'sentiment'] = 'n'
    df.loc[102, 'sentiment'] = 'n'
    df.loc[103, 'sentiment'] = 'p'
    df.loc[104, 'sentiment'] = 'p'
    df.loc[105, 'sentiment'] = 'n'
    df.loc[106, 'sentiment'] = 'p'
    df.loc[107, 'sentiment'] = 'p'
    df.loc[108, 'sentiment'] = 'n'
    df.loc[109, 'sentiment'] = 'p'
    df.loc[110, 'sentiment'] = 'n'
    df.loc[111, 'sentiment'] = 'p'
    df.loc[112, 'sentiment'] = 'p'
    df.loc[113, 'sentiment'] = 'p'
    df.loc[114, 'sentiment'] = 'n'
    df.loc[115, 'sentiment'] = 'n' #nn
    df.loc[116, 'sentiment'] = 'n'
    df.loc[117, 'sentiment'] = 'p'
    df.loc[118, 'sentiment'] = 'p'
    df.loc[119, 'sentiment'] = 'n'
    df.loc[120, 'sentiment'] = 'p'
    df.loc[121, 'sentiment'] = 'p'
    df.loc[122, 'sentiment'] = 'p'
    df.loc[123, 'sentiment'] = 'p'
    df.loc[124, 'sentiment'] = 'n'
    df.loc[125, 'sentiment'] = 'n'
    df.loc[126, 'sentiment'] = 'p'
    df.loc[127, 'sentiment'] = 'p'
    df.loc[128, 'sentiment'] = 'n'
    df.loc[129, 'sentiment'] = 'n'
    df.loc[130, 'sentiment'] = 'n'
    df.loc[131, 'sentiment'] = 'p'
    df.loc[132, 'sentiment'] = 'p'
    df.loc[133, 'sentiment'] = 'p'
    df.loc[134, 'sentiment'] = 'p'
    df.loc[135, 'sentiment'] = 'p'
    df.loc[136, 'sentiment'] = 'n'
    df.loc[137, 'sentiment'] = 'n'
    df.loc[138, 'sentiment'] = 'n'
    df.loc[139, 'sentiment'] = 'p'
    df.loc[140, 'sentiment'] = 'n'
    df.loc[141, 'sentiment'] = 'n'
    df.loc[142, 'sentiment'] = 'p' #n
    df.loc[143, 'sentiment'] = 'n'
    df.loc[144, 'sentiment'] = 'p'
    df.loc[145, 'sentiment'] = 'n'
    df.loc[146, 'sentiment'] = 'n'
    df.loc[147, 'sentiment'] = 'p'
    df.loc[148, 'sentiment'] = 'n'
    df.loc[149, 'sentiment'] = 'n'
    df.loc[150, 'sentiment'] = 'p'
    df.loc[151, 'sentiment'] = 'n'
    df.loc[152, 'sentiment'] = 'p'
    df.loc[153, 'sentiment'] = 'n'
    df.loc[154, 'sentiment'] = 'p' #n
    df.loc[155, 'sentiment'] = 'p'
    df.loc[156, 'sentiment'] = 'p'
    df.loc[157, 'sentiment'] = 'p'
    df.loc[158, 'sentiment'] = 'n'
    df.loc[159, 'sentiment'] = 'p'
    df.loc[160, 'sentiment'] = 'n'
    df.loc[161, 'sentiment'] = 'n'
    df.loc[162, 'sentiment'] = 'n'
    df.loc[163, 'sentiment'] = 'p'
    df.loc[164, 'sentiment'] = 'n'
    df.loc[165, 'sentiment'] = 'p'
    df.loc[166, 'sentiment'] = 'n'
    df.loc[167, 'sentiment'] = 'n'
    df.loc[168, 'sentiment'] = 'p'
    df.loc[169, 'sentiment'] = 'p'
    df.loc[170, 'sentiment'] = 'n'
    df.loc[171, 'sentiment'] = 'n'
    df.loc[172, 'sentiment'] = 'p' #n
    df.loc[173, 'sentiment'] = 'p'
    df.loc[174, 'sentiment'] = 'n'
    df.loc[175, 'sentiment'] = 'p'
    df.loc[176, 'sentiment'] = 'n'
    df.loc[177, 'sentiment'] = 'n'
    df.loc[178, 'sentiment'] = 'n'
    df.loc[179, 'sentiment'] = 'p'
    df.loc[180, 'sentiment'] = 'p'
    df.loc[181, 'sentiment'] = 'p'
    df.loc[182, 'sentiment'] = 'n'
    df.loc[183, 'sentiment'] = 'p' #n
    df.loc[184, 'sentiment'] = 'p'
    df.loc[185, 'sentiment'] = 'p' #n
    df.loc[186, 'sentiment'] = 'n'
    df.loc[187, 'sentiment'] = 'p'
    df.loc[188, 'sentiment'] = 'p'
    df.loc[189, 'sentiment'] = 'p'
    df.loc[190, 'sentiment'] = 'n'
    df.loc[191, 'sentiment'] = 'n'
    df.loc[192, 'sentiment'] = 'n'
    df.loc[193, 'sentiment'] = 'p'
    df.loc[194, 'sentiment'] = 'n'
    df.loc[195, 'sentiment'] = 'p'
    df.loc[196, 'sentiment'] = 'p'
    df.loc[197, 'sentiment'] = 'p'
    df.loc[198, 'sentiment'] = 'p'
    df.loc[199, 'sentiment'] = 'p'
    df.loc[200, 'sentiment'] = 'p'
    df.loc[201, 'sentiment'] = 'n'
    df.loc[202, 'sentiment'] = 'p' #n
    df.loc[203, 'sentiment'] = 'p' #n
    df.loc[204, 'sentiment'] = 'p'
    df.loc[205, 'sentiment'] = 'p'
    df.loc[206, 'sentiment'] = 'p'
    df.loc[207, 'sentiment'] = 'n'
    df.loc[208, 'sentiment'] = 'p'
    df.loc[209, 'sentiment'] = 'n'
    df.loc[210, 'sentiment'] = 'n'
    df.loc[211, 'sentiment'] = 'p'
    df.loc[212, 'sentiment'] = 'p'
    df.loc[213, 'sentiment'] = 'n'
    df.loc[214, 'sentiment'] = 'n'
    df.loc[215, 'sentiment'] = 'p'
    df.loc[216, 'sentiment'] = 'n'
    df.loc[217, 'sentiment'] = 'n'
    df.loc[218, 'sentiment'] = 'n'
    df.loc[219, 'sentiment'] = 'n'
    df.loc[220, 'sentiment'] = 'p'
    df.loc[221, 'sentiment'] = 'p'
    df.loc[222, 'sentiment'] = 'p'
    df.loc[223, 'sentiment'] = 'p'
    df.loc[224, 'sentiment'] = 'p'
    df.loc[225, 'sentiment'] = 'p' #n
    df.loc[226, 'sentiment'] = 'p' #n
    df.loc[227, 'sentiment'] = 'p'
    df.loc[228, 'sentiment'] = 'n'
    df.loc[229, 'sentiment'] = 'p'
    df.loc[230, 'sentiment'] = 'p'
    df.loc[231, 'sentiment'] = 'p' #nn
    df.loc[232, 'sentiment'] = 'n'
    df.loc[233, 'sentiment'] = 'p'
    df.loc[234, 'sentiment'] = 'p'
    df.loc[235, 'sentiment'] = 'p'
    df.loc[236, 'sentiment'] = 'n'
    df.loc[237, 'sentiment'] = 'n'
    df.loc[238, 'sentiment'] = 'p'
    df.loc[239, 'sentiment'] = 'p'
    df.loc[240, 'sentiment'] = 'p'
    df.loc[241, 'sentiment'] = 'p'
    df.loc[242, 'sentiment'] = 'n' #p?
    df.loc[243, 'sentiment'] = 'p' #n
    df.loc[244, 'sentiment'] = 'n'
    df.loc[245, 'sentiment'] = 'n'
    df.loc[246, 'sentiment'] = 'p'
    df.loc[247, 'sentiment'] = 'p'
    df.loc[248, 'sentiment'] = 'n'
    df.loc[249, 'sentiment'] = 'n'
    df.loc[250, 'sentiment'] = 'p'
    df.loc[251, 'sentiment'] = 'p'
    df.loc[252, 'sentiment'] = 'n'
    df.loc[253, 'sentiment'] = 'p'
    df.loc[254, 'sentiment'] = 'p' #n
    df.loc[255, 'sentiment'] = 'n'
    df.loc[256, 'sentiment'] = 'p'
    df.loc[257, 'sentiment'] = 'n'
    df.loc[258, 'sentiment'] = 'p'
    df.loc[259, 'sentiment'] = 'n' #p?
    df.loc[260, 'sentiment'] = 'p'
    df.loc[261, 'sentiment'] = 'n'
    df.loc[262, 'sentiment'] = 'p' #n
    df.loc[263, 'sentiment'] = 'p'
    df.loc[264, 'sentiment'] = 'n' #p?
    df.loc[265, 'sentiment'] = 'n'
    df.loc[266, 'sentiment'] = 'p'
    df.loc[267, 'sentiment'] = 'n'
    df.loc[268, 'sentiment'] = 'p'
    df.loc[269, 'sentiment'] = 'p'
    df.loc[270, 'sentiment'] = 'p'
    df.loc[271, 'sentiment'] = 'p' #n
    df.loc[272, 'sentiment'] = 'n'
    df.loc[273, 'sentiment'] = 'p'
    df.loc[274, 'sentiment'] = 'p'
    df.loc[275, 'sentiment'] = 'p'
    df.loc[276, 'sentiment'] = 'n'
    df.loc[277, 'sentiment'] = 'n'
    df.loc[278, 'sentiment'] = 'p'
    df.loc[279, 'sentiment'] = 'n'
    df.loc[280, 'sentiment'] = 'n'
    df.loc[281, 'sentiment'] = 'n'
    df.loc[282, 'sentiment'] = 'n'
    df.loc[283, 'sentiment'] = 'n'
    df.loc[284, 'sentiment'] = 'p'
    df.loc[285, 'sentiment'] = 'n'
    df.loc[286, 'sentiment'] = 'p'
    df.loc[287, 'sentiment'] = 'p'
    df.loc[288, 'sentiment'] = 'p'
    df.loc[289, 'sentiment'] = 'p'
    df.loc[290, 'sentiment'] = 'n'
    df.loc[291, 'sentiment'] = 'n'
    df.loc[292, 'sentiment'] = 'p'
    df.loc[293, 'sentiment'] = 'n'
    df.loc[294, 'sentiment'] = 'n'
    df.loc[295, 'sentiment'] = 'n'
    df.loc[296, 'sentiment'] = 'n'
    df.loc[297, 'sentiment'] = 'n'
    df.loc[298, 'sentiment'] = 'p'
    df.loc[299, 'sentiment'] = 'p'
    df.loc[300, 'sentiment'] = 'p'
    df.loc[301, 'sentiment'] = 'n'
    df.loc[302, 'sentiment'] = 'n'
    df.loc[303, 'sentiment'] = 'n'
    df.loc[304, 'sentiment'] = 'p'
    df.loc[305, 'sentiment'] = 'p'
    df.loc[306, 'sentiment'] = 'p' #n
    df.loc[307, 'sentiment'] = 'n'
    df.loc[308, 'sentiment'] = 'n'
    df.loc[309, 'sentiment'] = 'n'
    df.loc[310, 'sentiment'] = 'p' #n
    df.loc[311, 'sentiment'] = 'n'
    df.loc[312, 'sentiment'] = 'p' #n
    df.loc[313, 'sentiment'] = 'p'  #n
    df.loc[314, 'sentiment'] = 'p'
    df.loc[315, 'sentiment'] = 'n'
    df.loc[316, 'sentiment'] = 'n'
    df.loc[317, 'sentiment'] = 'p' #n
    df.loc[318, 'sentiment'] = 'n'
    df.loc[319, 'sentiment'] = 'n'
    df.loc[320, 'sentiment'] = 'p'
    df.loc[321, 'sentiment'] = 'p'
    df.loc[322, 'sentiment'] = 'p'
    df.loc[323, 'sentiment'] = 'n'
    df.loc[324, 'sentiment'] = 'n'
    df.loc[325, 'sentiment'] = 'n'
    df.loc[326, 'sentiment'] = 'n'
    df.loc[327, 'sentiment'] = 'n'
    df.loc[328, 'sentiment'] = 'p'
    df.loc[329, 'sentiment'] = 'p'
    df.loc[330, 'sentiment'] = 'p'
    df.loc[331, 'sentiment'] = 'p'
    df.loc[332, 'sentiment'] = 'p' #n
    df.loc[333, 'sentiment'] = 'p' #n
    df.loc[334, 'sentiment'] = 'n'
    df.loc[335, 'sentiment'] = 'p' #n
    df.loc[336, 'sentiment'] = 'p'
    df.loc[337, 'sentiment'] = 'n'
    df.loc[338, 'sentiment'] = 'p'
    df.loc[339, 'sentiment'] = 'p'
    df.loc[340, 'sentiment'] = 'p' #n
    df.loc[341, 'sentiment'] = 'p'
    df.loc[342, 'sentiment'] = 'n'
    df.loc[343, 'sentiment'] = 'p'
    df.loc[344, 'sentiment'] = 'n' #p?
    df.loc[345, 'sentiment'] = 'p' #n
    df.loc[346, 'sentiment'] = 'p'
    df.loc[347, 'sentiment'] = 'p' #n
    df.loc[348, 'sentiment'] = 'p'
    df.loc[349, 'sentiment'] = 'p'
    df.loc[350, 'sentiment'] = 'p' #n
    df.loc[351, 'sentiment'] = 'n'
    df.loc[352, 'sentiment'] = 'p'
    df.loc[353, 'sentiment'] = 'p'
    df.loc[354, 'sentiment'] = 'p' #n
    df.loc[355, 'sentiment'] = 'n'
    df.loc[356, 'sentiment'] = 'p'
    df.loc[357, 'sentiment'] = 'p' #n
    df.loc[358, 'sentiment'] = 'n'
    df.loc[359, 'sentiment'] = 'n'
    df.loc[360, 'sentiment'] = 'p' #n
    df.loc[361, 'sentiment'] = 'n' #sarcasm
    df.loc[362, 'sentiment'] = 'p'
    df.loc[363, 'sentiment'] = 'n'
    df.loc[364, 'sentiment'] = 'p' #n
    df.loc[365, 'sentiment'] = 'n'
    df.loc[366, 'sentiment'] = 'n'
    df.loc[367, 'sentiment'] = 'n'
    df.loc[368, 'sentiment'] = 'p' #n
    df.loc[369, 'sentiment'] = 'p'
    df.loc[370, 'sentiment'] = 'p' #n
    df.loc[371, 'sentiment'] = 'p' #n
    df.loc[372, 'sentiment'] = 'n'
    df.loc[373, 'sentiment'] = 'n'
    df.loc[374, 'sentiment'] = 'n'
    df.loc[375, 'sentiment'] = 'p'
    df.loc[376, 'sentiment'] = 'p' #n
    df.loc[377, 'sentiment'] = 'p'
    df.loc[378, 'sentiment'] = 'n'
    df.loc[379, 'sentiment'] = 'n' #n?
    df.loc[380, 'sentiment'] = 'p' #n
    df.loc[381, 'sentiment'] = 'p' #n
    df.loc[382, 'sentiment'] = 'p'

    df.to_csv("ag_news_383_labeled_final.csv", index=False, encoding="utf-8-sig")

#apply_sentiment_and_save()

# LLM IMDb
#def create_imdb_standard_sample():
   # df_sample, _ = sample_imdb()
    #df_sample.head(385).to_csv("imdb_standard_sample_385.csv", index=True, encoding="utf-8-sig")

#create_imdb_standard_sample()

def apply_imdb_sentiment_and_save():
    df = pd.read_csv("imdb_standard_sample_385.csv", index_col=0)
    df['sentiment'] = ""


def apply_imdb_sentiment_and_save():
    df = pd.read_csv("imdb_standard_sample_385.csv", index_col=0)
    df['sentiment'] = ""
    df.loc[0, 'sentiment'] = 'n'
    df.loc[1, 'sentiment'] = 'n'
    df.loc[2, 'sentiment'] = 'n'
    df.loc[3, 'sentiment'] = 'n'
    df.loc[4, 'sentiment'] = 'n'
    df.loc[5, 'sentiment'] = 'n'
    df.loc[6, 'sentiment'] = 'n'
    df.loc[7, 'sentiment'] = 'n'
    df.loc[8, 'sentiment'] = 'n'
    df.loc[9, 'sentiment'] = 'n'
    df.loc[10, 'sentiment'] = 'n'
    df.loc[11, 'sentiment'] = 'n'
    df.loc[12, 'sentiment'] = 'n'
    df.loc[13, 'sentiment'] = 'n'
    df.loc[14, 'sentiment'] = 'n'
    df.loc[15, 'sentiment'] = 'n'
    df.loc[16, 'sentiment'] = 'n'
    df.loc[17, 'sentiment'] = 'n'
    df.loc[18, 'sentiment'] = 'n'
    df.loc[19, 'sentiment'] = 'n'
    df.loc[20, 'sentiment'] = 'n'
    df.loc[21, 'sentiment'] = 'n'
    df.loc[22, 'sentiment'] = 'n'
    df.loc[23, 'sentiment'] = 'n'
    df.loc[24, 'sentiment'] = 'n'
    df.loc[25, 'sentiment'] = 'n'
    df.loc[26, 'sentiment'] = 'n'
    df.loc[27, 'sentiment'] = 'n'
    df.loc[28, 'sentiment'] = 'n'
    df.loc[29, 'sentiment'] = 'n'
    df.loc[30, 'sentiment'] = 'n'
    df.loc[31, 'sentiment'] = 'n'
    df.loc[32, 'sentiment'] = 'n'
    df.loc[33, 'sentiment'] = 'n'
    df.loc[34, 'sentiment'] = 'n'
    df.loc[35, 'sentiment'] = 'n'
    df.loc[36, 'sentiment'] = 'n'
    df.loc[37, 'sentiment'] = 'n'
    df.loc[38, 'sentiment'] = 'n'
    df.loc[39, 'sentiment'] = 'n'
    df.loc[40, 'sentiment'] = 'n'
    df.loc[41, 'sentiment'] = 'n'
    df.loc[42, 'sentiment'] = 'n'
    df.loc[43, 'sentiment'] = 'n'
    df.loc[44, 'sentiment'] = 'n'
    df.loc[45, 'sentiment'] = 'n'
    df.loc[46, 'sentiment'] = 'n'
    df.loc[47, 'sentiment'] = 'n'
    df.loc[48, 'sentiment'] = 'n'
    df.loc[49, 'sentiment'] = 'n'
    df.loc[50, 'sentiment'] = 'n'
    df.loc[51, 'sentiment'] = 'n'
    df.loc[52, 'sentiment'] = 'n'
    df.loc[53, 'sentiment'] = 'n'
    df.loc[54, 'sentiment'] = 'n'
    df.loc[55, 'sentiment'] = 'n'
    df.loc[56, 'sentiment'] = 'n'
    df.loc[57, 'sentiment'] = 'n'
    df.loc[58, 'sentiment'] = 'n'
    df.loc[59, 'sentiment'] = 'n'
    df.loc[60, 'sentiment'] = 'n'
    df.loc[61, 'sentiment'] = 'n'
    df.loc[62, 'sentiment'] = 'n'
    df.loc[63, 'sentiment'] = 'n'
    df.loc[64, 'sentiment'] = 'n'
    df.loc[65, 'sentiment'] = 'n'
    df.loc[66, 'sentiment'] = 'n'
    df.loc[67, 'sentiment'] = 'n'
    df.loc[68, 'sentiment'] = 'n'
    df.loc[69, 'sentiment'] = 'n'
    df.loc[70, 'sentiment'] = 'n'
    df.loc[71, 'sentiment'] = 'n'
    df.loc[72, 'sentiment'] = 'n'
    df.loc[73, 'sentiment'] = 'n'
    df.loc[74, 'sentiment'] = 'n'
    df.loc[75, 'sentiment'] = 'n'
    df.loc[76, 'sentiment'] = 'n'
    df.loc[77, 'sentiment'] = 'n'
    df.loc[78, 'sentiment'] = 'n'
    df.loc[79, 'sentiment'] = 'n'
    df.loc[80, 'sentiment'] = 'n'
    df.loc[81, 'sentiment'] = 'n'
    df.loc[82, 'sentiment'] = 'n'
    df.loc[83, 'sentiment'] = 'n'
    df.loc[84, 'sentiment'] = 'n'
    df.loc[85, 'sentiment'] = 'n'
    df.loc[86, 'sentiment'] = 'n'
    df.loc[87, 'sentiment'] = 'n'
    df.loc[88, 'sentiment'] = 'n'
    df.loc[89, 'sentiment'] = 'n'
    df.loc[90, 'sentiment'] = 'n'
    df.loc[91, 'sentiment'] = 'n'
    df.loc[92, 'sentiment'] = 'n'
    df.loc[93, 'sentiment'] = 'n'
    df.loc[94, 'sentiment'] = 'n'
    df.loc[95, 'sentiment'] = 'n'
    df.loc[96, 'sentiment'] = 'n'
    df.loc[97, 'sentiment'] = 'n'
    df.loc[98, 'sentiment'] = 'n'
    df.loc[99, 'sentiment'] = 'n'
    df.loc[100, 'sentiment'] = 'n'
    df.loc[101, 'sentiment'] = 'n'
    df.loc[102, 'sentiment'] = 'n'
    df.loc[103, 'sentiment'] = 'n'
    df.loc[104, 'sentiment'] = 'n'
    df.loc[105, 'sentiment'] = 'n'
    df.loc[106, 'sentiment'] = 'n'
    df.loc[107, 'sentiment'] = 'n'
    df.loc[108, 'sentiment'] = 'n'
    df.loc[109, 'sentiment'] = 'n'
    df.loc[110, 'sentiment'] = 'n'
    df.loc[111, 'sentiment'] = 'n'
    df.loc[112, 'sentiment'] = 'n'
    df.loc[113, 'sentiment'] = 'n'
    df.loc[114, 'sentiment'] = 'n'
    df.loc[115, 'sentiment'] = 'n'
    df.loc[116, 'sentiment'] = 'n'
    df.loc[117, 'sentiment'] = 'n'
    df.loc[118, 'sentiment'] = 'n'
    df.loc[119, 'sentiment'] = 'n'
    df.loc[120, 'sentiment'] = 'n'
    df.loc[121, 'sentiment'] = 'n'
    df.loc[122, 'sentiment'] = 'n'
    df.loc[123, 'sentiment'] = 'n'
    df.loc[124, 'sentiment'] = 'n'
    df.loc[125, 'sentiment'] = 'n'
    df.loc[126, 'sentiment'] = 'n'
    df.loc[127, 'sentiment'] = 'n'
    df.loc[128, 'sentiment'] = 'n'
    df.loc[129, 'sentiment'] = 'n'
    df.loc[130, 'sentiment'] = 'n'
    df.loc[131, 'sentiment'] = 'n'
    df.loc[132, 'sentiment'] = 'n'
    df.loc[133, 'sentiment'] = 'n'
    df.loc[134, 'sentiment'] = 'n'
    df.loc[135, 'sentiment'] = 'n'
    df.loc[136, 'sentiment'] = 'n'
    df.loc[137, 'sentiment'] = 'n'
    df.loc[138, 'sentiment'] = 'n'
    df.loc[139, 'sentiment'] = 'n'
    df.loc[140, 'sentiment'] = 'n'
    df.loc[141, 'sentiment'] = 'n'
    df.loc[142, 'sentiment'] = 'n'
    df.loc[143, 'sentiment'] = 'n'
    df.loc[144, 'sentiment'] = 'n'
    df.loc[145, 'sentiment'] = 'n'
    df.loc[146, 'sentiment'] = 'n'
    df.loc[147, 'sentiment'] = 'n'
    df.loc[148, 'sentiment'] = 'n'
    df.loc[149, 'sentiment'] = 'n'
    df.loc[150, 'sentiment'] = 'n'
    df.loc[151, 'sentiment'] = 'n'
    df.loc[152, 'sentiment'] = 'n'
    df.loc[153, 'sentiment'] = 'n'
    df.loc[154, 'sentiment'] = 'n'
    df.loc[155, 'sentiment'] = 'n'
    df.loc[156, 'sentiment'] = 'n'
    df.loc[157, 'sentiment'] = 'n'
    df.loc[158, 'sentiment'] = 'n'
    df.loc[159, 'sentiment'] = 'n'
    df.loc[160, 'sentiment'] = 'n'
    df.loc[161, 'sentiment'] = 'n'
    df.loc[162, 'sentiment'] = 'n'
    df.loc[163, 'sentiment'] = 'n'
    df.loc[164, 'sentiment'] = 'n'
    df.loc[165, 'sentiment'] = 'n'
    df.loc[166, 'sentiment'] = 'n'
    df.loc[167, 'sentiment'] = 'n'
    df.loc[168, 'sentiment'] = 'n'
    df.loc[169, 'sentiment'] = 'n'
    df.loc[170, 'sentiment'] = 'n'
    df.loc[171, 'sentiment'] = 'n'
    df.loc[172, 'sentiment'] = 'n'
    df.loc[173, 'sentiment'] = 'n'
    df.loc[174, 'sentiment'] = 'n'
    df.loc[175, 'sentiment'] = 'n'
    df.loc[176, 'sentiment'] = 'n'
    df.loc[177, 'sentiment'] = 'n'
    df.loc[178, 'sentiment'] = 'n'
    df.loc[179, 'sentiment'] = 'n'
    df.loc[180, 'sentiment'] = 'n'
    df.loc[181, 'sentiment'] = 'n'
    df.loc[182, 'sentiment'] = 'n'
    df.loc[183, 'sentiment'] = 'n'
    df.loc[184, 'sentiment'] = 'n'
    df.loc[185, 'sentiment'] = 'n'
    df.loc[186, 'sentiment'] = 'n'
    df.loc[187, 'sentiment'] = 'n'
    df.loc[188, 'sentiment'] = 'n'
    df.loc[189, 'sentiment'] = 'n'
    df.loc[190, 'sentiment'] = 'n'
    df.loc[191, 'sentiment'] = 'n'
    df.loc[192, 'sentiment'] = 'p'
    df.loc[193, 'sentiment'] = 'p'
    df.loc[194, 'sentiment'] = 'p'
    df.loc[195, 'sentiment'] = 'p'
    df.loc[196, 'sentiment'] = 'p'
    df.loc[197, 'sentiment'] = 'p'
    df.loc[198, 'sentiment'] = 'p'
    df.loc[199, 'sentiment'] = 'p'
    df.loc[200, 'sentiment'] = 'p'
    df.loc[201, 'sentiment'] = 'p'
    df.loc[202, 'sentiment'] = 'p'
    df.loc[203, 'sentiment'] = 'p'
    df.loc[204, 'sentiment'] = 'p'
    df.loc[205, 'sentiment'] = 'p'
    df.loc[206, 'sentiment'] = 'p'
    df.loc[207, 'sentiment'] = 'p'
    df.loc[208, 'sentiment'] = 'p'
    df.loc[209, 'sentiment'] = 'p'
    df.loc[210, 'sentiment'] = 'p'
    df.loc[211, 'sentiment'] = 'p'
    df.loc[212, 'sentiment'] = 'p'
    df.loc[213, 'sentiment'] = 'p'
    df.loc[214, 'sentiment'] = 'p'
    df.loc[215, 'sentiment'] = 'p'
    df.loc[216, 'sentiment'] = 'p'
    df.loc[217, 'sentiment'] = 'p'
    df.loc[218, 'sentiment'] = 'p'
    df.loc[219, 'sentiment'] = 'p'
    df.loc[220, 'sentiment'] = 'p'
    df.loc[221, 'sentiment'] = 'p'
    df.loc[222, 'sentiment'] = 'p'
    df.loc[223, 'sentiment'] = 'p'
    df.loc[224, 'sentiment'] = 'p'
    df.loc[225, 'sentiment'] = 'p'
    df.loc[226, 'sentiment'] = 'p'
    df.loc[227, 'sentiment'] = 'p'
    df.loc[228, 'sentiment'] = 'p'
    df.loc[229, 'sentiment'] = 'p'
    df.loc[230, 'sentiment'] = 'p'
    df.loc[231, 'sentiment'] = 'p'
    df.loc[232, 'sentiment'] = 'p'
    df.loc[233, 'sentiment'] = 'p'
    df.loc[234, 'sentiment'] = 'p'
    df.loc[235, 'sentiment'] = 'p'
    df.loc[236, 'sentiment'] = 'p'
    df.loc[237, 'sentiment'] = 'p'
    df.loc[238, 'sentiment'] = 'p'
    df.loc[239, 'sentiment'] = 'p'
    df.loc[240, 'sentiment'] = 'p'
    df.loc[241, 'sentiment'] = 'p'
    df.loc[242, 'sentiment'] = 'p'
    df.loc[243, 'sentiment'] = 'p'
    df.loc[244, 'sentiment'] = 'p'
    df.loc[245, 'sentiment'] = 'p'
    df.loc[246, 'sentiment'] = 'p'
    df.loc[247, 'sentiment'] = 'p'
    df.loc[248, 'sentiment'] = 'p'
    df.loc[249, 'sentiment'] = 'p'
    df.loc[250, 'sentiment'] = 'p'
    df.loc[251, 'sentiment'] = 'p'
    df.loc[252, 'sentiment'] = 'p'
    df.loc[253, 'sentiment'] = 'p'
    df.loc[254, 'sentiment'] = 'p'
    df.loc[255, 'sentiment'] = 'p'
    df.loc[256, 'sentiment'] = 'p'
    df.loc[257, 'sentiment'] = 'p'
    df.loc[258, 'sentiment'] = 'p'
    df.loc[259, 'sentiment'] = 'p'
    df.loc[260, 'sentiment'] = 'p'
    df.loc[261, 'sentiment'] = 'p'
    df.loc[262, 'sentiment'] = 'p'
    df.loc[263, 'sentiment'] = 'p'
    df.loc[264, 'sentiment'] = 'p'
    df.loc[265, 'sentiment'] = 'p'
    df.loc[266, 'sentiment'] = 'p'
    df.loc[267, 'sentiment'] = 'p'
    df.loc[268, 'sentiment'] = 'p'
    df.loc[269, 'sentiment'] = 'p'
    df.loc[270, 'sentiment'] = 'p'
    df.loc[271, 'sentiment'] = 'p'
    df.loc[272, 'sentiment'] = 'p'
    df.loc[273, 'sentiment'] = 'p'
    df.loc[274, 'sentiment'] = 'p'
    df.loc[275, 'sentiment'] = 'p'
    df.loc[276, 'sentiment'] = 'p'
    df.loc[277, 'sentiment'] = 'p'
    df.loc[278, 'sentiment'] = 'p'
    df.loc[279, 'sentiment'] = 'p'
    df.loc[280, 'sentiment'] = 'p'
    df.loc[281, 'sentiment'] = 'p'
    df.loc[282, 'sentiment'] = 'p'
    df.loc[283, 'sentiment'] = 'p'
    df.loc[284, 'sentiment'] = 'p'
    df.loc[285, 'sentiment'] = 'p'
    df.loc[286, 'sentiment'] = 'p'
    df.loc[287, 'sentiment'] = 'p'
    df.loc[288, 'sentiment'] = 'p'
    df.loc[289, 'sentiment'] = 'p'
    df.loc[290, 'sentiment'] = 'p'
    df.loc[291, 'sentiment'] = 'p'
    df.loc[292, 'sentiment'] = 'p'
    df.loc[293, 'sentiment'] = 'p'
    df.loc[294, 'sentiment'] = 'p'
    df.loc[295, 'sentiment'] = 'p'
    df.loc[296, 'sentiment'] = 'p'
    df.loc[297, 'sentiment'] = 'p'
    df.loc[298, 'sentiment'] = 'p'
    df.loc[299, 'sentiment'] = 'p'
    df.loc[300, 'sentiment'] = 'p'
    df.loc[301, 'sentiment'] = 'p'
    df.loc[302, 'sentiment'] = 'p'
    df.loc[303, 'sentiment'] = 'p'
    df.loc[304, 'sentiment'] = 'p'
    df.loc[305, 'sentiment'] = 'p'
    df.loc[306, 'sentiment'] = 'p'
    df.loc[307, 'sentiment'] = 'p'
    df.loc[308, 'sentiment'] = 'p'
    df.loc[309, 'sentiment'] = 'p'
    df.loc[310, 'sentiment'] = 'p'
    df.loc[311, 'sentiment'] = 'p'
    df.loc[312, 'sentiment'] = 'p'
    df.loc[313, 'sentiment'] = 'p'
    df.loc[314, 'sentiment'] = 'p'
    df.loc[315, 'sentiment'] = 'p'
    df.loc[316, 'sentiment'] = 'p'
    df.loc[317, 'sentiment'] = 'p'
    df.loc[318, 'sentiment'] = 'p'
    df.loc[319, 'sentiment'] = 'p'
    df.loc[320, 'sentiment'] = 'p'
    df.loc[321, 'sentiment'] = 'p'
    df.loc[322, 'sentiment'] = 'p'
    df.loc[323, 'sentiment'] = 'p'
    df.loc[324, 'sentiment'] = 'p'
    df.loc[325, 'sentiment'] = 'p'
    df.loc[326, 'sentiment'] = 'p'
    df.loc[327, 'sentiment'] = 'p'
    df.loc[328, 'sentiment'] = 'p'
    df.loc[329, 'sentiment'] = 'p'
    df.loc[330, 'sentiment'] = 'p'
    df.loc[331, 'sentiment'] = 'p'
    df.loc[332, 'sentiment'] = 'p'
    df.loc[333, 'sentiment'] = 'p'
    df.loc[334, 'sentiment'] = 'p'
    df.loc[335, 'sentiment'] = 'p'
    df.loc[336, 'sentiment'] = 'p'
    df.loc[337, 'sentiment'] = 'p'
    df.loc[338, 'sentiment'] = 'p'
    df.loc[339, 'sentiment'] = 'p'
    df.loc[340, 'sentiment'] = 'p'
    df.loc[341, 'sentiment'] = 'p'
    df.loc[342, 'sentiment'] = 'p'
    df.loc[343, 'sentiment'] = 'p'
    df.loc[344, 'sentiment'] = 'p'
    df.loc[345, 'sentiment'] = 'p'
    df.loc[346, 'sentiment'] = 'p'
    df.loc[347, 'sentiment'] = 'p'
    df.loc[348, 'sentiment'] = 'p'
    df.loc[349, 'sentiment'] = 'p'
    df.loc[350, 'sentiment'] = 'p'
    df.loc[351, 'sentiment'] = 'p'
    df.loc[352, 'sentiment'] = 'p'
    df.loc[353, 'sentiment'] = 'p'
    df.loc[354, 'sentiment'] = 'p'
    df.loc[355, 'sentiment'] = 'p'
    df.loc[356, 'sentiment'] = 'p'
    df.loc[357, 'sentiment'] = 'p'
    df.loc[358, 'sentiment'] = 'p'
    df.loc[359, 'sentiment'] = 'p'
    df.loc[360, 'sentiment'] = 'p'
    df.loc[361, 'sentiment'] = 'p'
    df.loc[362, 'sentiment'] = 'p'
    df.loc[363, 'sentiment'] = 'p'
    df.loc[364, 'sentiment'] = 'p'
    df.loc[365, 'sentiment'] = 'p'
    df.loc[366, 'sentiment'] = 'p'
    df.loc[367, 'sentiment'] = 'p'
    df.loc[368, 'sentiment'] = 'p'
    df.loc[369, 'sentiment'] = 'p'
    df.loc[370, 'sentiment'] = 'p'
    df.loc[371, 'sentiment'] = 'p'
    df.loc[372, 'sentiment'] = 'p'
    df.loc[373, 'sentiment'] = 'p'
    df.loc[374, 'sentiment'] = 'p'
    df.loc[375, 'sentiment'] = 'p'
    df.loc[376, 'sentiment'] = 'p'
    df.loc[377, 'sentiment'] = 'p'
    df.loc[378, 'sentiment'] = 'p'
    df.loc[379, 'sentiment'] = 'p'
    df.loc[380, 'sentiment'] = 'p'
    df.loc[381, 'sentiment'] = 'p'
    df.loc[382, 'sentiment'] = 'p'
    df.loc[383, 'sentiment'] = 'p'
    df.loc[384, 'sentiment'] = 'p'




    df.to_csv("imdb_385_labeled_final.csv", index=False, encoding="utf-8-sig")

#apply_imdb_sentiment_and_save()













