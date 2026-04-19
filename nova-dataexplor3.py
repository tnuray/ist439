import customtkinter as ctk
import pandas as pd
import numpy as np
from tkinter import messagebox, filedialog
from tabulate import tabulate
import fasttext


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
        self.save_button.pack(pady=5, anchor="w", padx=20)

        # Summary Button (bottom right)
        self.summary_button = ctk.CTkButton(self, text="📊 Summary Statistics",
                                            command=self.show_summary,
                                            state="disabled", fg_color="gray")
        self.summary_button.pack(pady=5, anchor="e", padx=20)

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


if __name__ == "__main__":
    app = DataExplor()
    app.mainloop()