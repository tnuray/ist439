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
            else:
                df, label_col = sample_imdb()
                filename = "imdb"

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
        if self.current_df is None:
            return

        raw_paths = {
            "yelp": "yelp_stars_fixed.csv",
            "ag_news": "indirilen_veriler/ag_news.csv",
            "imdb": "indirilen_veriler/imdb.csv"
        }
        raw_label_cols = {
            "yelp": "new_label_star",
            "ag_news": "label",
            "imdb": "label"
        }

        try:
            df_raw = pd.read_csv(raw_paths[self.current_filename])
            raw_label = raw_label_cols[self.current_filename]

            summary = f"\n\n--- Full Dataset Summary: {self.current_filename} ---\n"
            summary += f"Total rows: {len(df_raw):,}\n\n"
            summary += "Label Distribution:\n"
            summary += df_raw[raw_label].value_counts().sort_index().to_string()

            self.text_display.insert("end", summary)

        except FileNotFoundError:
            self.text_display.insert("end", "\n[Full Dataset] File not found.\n")

    def display_final_labeled_data(self):
        try:
            df = pd.read_csv("yelp_400_labeled_final.csv")
            display_df = df[['new_label_star', 'sentiment', 'text']].copy()
            display_df['text'] = display_df['text'].astype(str).str.slice(0, 55) + "..."
            table = tabulate(display_df.head(400), headers='keys', tablefmt='psql', showindex=True)

            self.text_display.delete("1.0", "end")
            self.text_display.insert("end", "--- FINAL LABELED DATASET (YELP) ---\n\n")
            self.text_display.insert("end", table)

            self.status_label.configure(text="Status: yelp_400_labeled_final.csv loaded! ✨", text_color="#00FF00")
            self.save_button.configure(state="normal", fg_color="#1f6aa5")
        except Exception as e:
            print(f"Error: Could not find the file yelp_400_labeled_final.csv - {e}")
            self.status_label.configure(text="Status: Labeled file not found!", text_color="red")

    def run_llm_prediction(self):
        labeled_file = "yelp_400_labeled_final.csv"
        test_file = "yelp_standard_sample_400.csv"

        if not os.path.exists(labeled_file):
            messagebox.showerror("Error", "Please organize the data using Summary Statistics first!")
            return

        try:
            self.status_label.configure(text="Status: Learning from your labels...", text_color="orange")
            self.update()

            train_df = pd.read_csv(labeled_file)
            model = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', LogisticRegression())
            ])

            model.fit(train_df['text'].astype(str), train_df['sentiment'])
            test_df = pd.read_csv(test_file)
            predictions = model.predict(test_df['text'].astype(str))

            self.text_display.delete("1.0", "end")
            self.text_display.insert("end", "📊 --- PREDICTION BASED ON YOUR ORGANIZED DATA --- 📊\n")
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

if __name__ == "__main__":
    app = DataExplor()
    app.display_final_labeled_data()
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
    df.loc[174, 'sentiment'] = 'n'  # n
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
    df.loc[192, 'sentiment'] = 'n'
    df.loc[193, 'sentiment'] = 'n'
    df.loc[194, 'sentiment'] = 'n'
    df.loc[195, 'sentiment'] = 'n'
    df.loc[196, 'sentiment'] = 'n'
    df.loc[197, 'sentiment'] = 'n'
    df.loc[198, 'sentiment'] = 'n'
    df.loc[199, 'sentiment'] = 'n'
    df.loc[200, 'sentiment'] = 'n'
    df.loc[201, 'sentiment'] = 'n'
    df.loc[202, 'sentiment'] = 'n'
    df.loc[203, 'sentiment'] = 'n'
    df.loc[204, 'sentiment'] = 'n'
    df.loc[205, 'sentiment'] = 'n'
    df.loc[206, 'sentiment'] = 'n'
    df.loc[207, 'sentiment'] = 'n'
    df.loc[208, 'sentiment'] = 'n'
    df.loc[209, 'sentiment'] = 'n'
    df.loc[210, 'sentiment'] = 'n'
    df.loc[211, 'sentiment'] = 'n'
    df.loc[212, 'sentiment'] = 'n'
    df.loc[213, 'sentiment'] = 'n'
    df.loc[214, 'sentiment'] = 'n'
    df.loc[215, 'sentiment'] = 'n'
    df.loc[216, 'sentiment'] = 'n'
    df.loc[217, 'sentiment'] = 'n'
    df.loc[218, 'sentiment'] = 'n'
    df.loc[219, 'sentiment'] = 'n'
    df.loc[220, 'sentiment'] = 'n'
    df.loc[221, 'sentiment'] = 'n'
    df.loc[222, 'sentiment'] = 'n'
    df.loc[223, 'sentiment'] = 'n'
    df.loc[224, 'sentiment'] = 'n'
    df.loc[225, 'sentiment'] = 'n'
    df.loc[226, 'sentiment'] = 'n'
    df.loc[227, 'sentiment'] = 'n'
    df.loc[228, 'sentiment'] = 'n'
    df.loc[229, 'sentiment'] = 'n'
    df.loc[230, 'sentiment'] = 'n'
    df.loc[231, 'sentiment'] = 'n'
    df.loc[232, 'sentiment'] = 'n'
    df.loc[233, 'sentiment'] = 'n'
    df.loc[234, 'sentiment'] = 'n'
    df.loc[235, 'sentiment'] = 'n'
    df.loc[236, 'sentiment'] = 'n'
    df.loc[237, 'sentiment'] = 'n'
    df.loc[238, 'sentiment'] = 'n'
    df.loc[239, 'sentiment'] = 'n'
    df.loc[240, 'sentiment'] = 'n'
    df.loc[241, 'sentiment'] = 'n'
    df.loc[242, 'sentiment'] = 'n'
    df.loc[243, 'sentiment'] = 'n'
    df.loc[244, 'sentiment'] = 'n'
    df.loc[245, 'sentiment'] = 'n'
    df.loc[246, 'sentiment'] = 'n'
    df.loc[247, 'sentiment'] = 'n'
    df.loc[248, 'sentiment'] = 'n'
    df.loc[249, 'sentiment'] = 'n'
    df.loc[250, 'sentiment'] = 'n'
    df.loc[251, 'sentiment'] = 'n'  # n
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

    df.to_csv("yelp_400_labeled_final.csv", index=False, encoding="utf-8-sig")

# apply_sentiment_and_save()

# LLM
