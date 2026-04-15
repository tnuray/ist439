import customtkinter as ctk
import pandas as pd
from tkinter import messagebox
from tabulate import tabulate

# Appearance Settings
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class DataExplor(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Data Explorer 🦄")
        self.geometry("950x750")

        # --- UI ELEMENTS ---
        self.label_title = ctk.CTkLabel(self, text="Data Explorer Bot 🤖", font=("Helvetica", 24, "bold"))
        self.label_title.pack(pady=20)

        self.label_info = ctk.CTkLabel(self, text="Select a dataset to view the processed samples:",
                                       font=("Helvetica", 14))
        self.label_info.pack(pady=5)

        # Buttons Panel
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=15)

        # Dataset Buttons

        ctk.CTkButton(self.button_frame, text="Yelp Reviews",
                      command=lambda: self.load_data("yelp_processed_50_final.csv", "new_label_star")).grid(row=0,
                                                                                                            column=0,
                                                                                                            padx=10,
                                                                                                            pady=10)

        ctk.CTkButton(self.button_frame, text="AG News",
                      command=lambda: self.load_data("ag_news_processed_50_final.csv", "label")).grid(row=0, column=1,
                                                                                                      padx=10, pady=10)

        ctk.CTkButton(self.button_frame, text="IMDB Reviews",
                      command=lambda: self.load_data("imdb_processed_50_final.csv", "label")).grid(row=0, column=2,
                                                                                                   padx=10, pady=10)

        # Display Area (Consolas font ensures table alignment)
        self.text_display = ctk.CTkTextbox(self, width=900, height=450, font=("Consolas", 12))
        self.text_display.pack(pady=20, padx=20)
        self.text_display.insert("0.0", "System: Ready. Please select a dataset to start exploration... ✨\n")

        # Status Bar
        self.status_label = ctk.CTkLabel(self, text="Status: Idle", font=("Helvetica", 12), text_color="gray")
        self.status_label.pack(side="bottom", pady=10)

    def load_data(self, file_name, label_col):
        try:
            # 1. Load the CSV file
            df = pd.read_csv(file_name)

            # 2. Select specific columns (Label, Category, Text)
            # 'new_column' is the manual label I created in my previous code.
            display_df = df[[label_col, 'new_column', 'text']].copy()

            # 3. Shorten text for better table formatting
            display_df['text'] = display_df['text'].astype(str).str.slice(0, 70) + "..."

            # 4. Generate the formatted table using Tabulate
            table = tabulate(display_df, headers='keys', tablefmt='psql', showindex=True)

            # 5. Update the display
            self.text_display.delete("1.0", "end")
            self.text_display.insert("end", f"--- Exploration Results: {file_name} ---\n\n")
            self.text_display.insert("end", table)

            # Update Status
            self.status_label.configure(text=f"Status: {file_name} loaded successfully.", text_color="#00FF00")

        except FileNotFoundError:
            messagebox.showerror("Error", f"File '{file_name}' not found. Please check your project folder.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    app = DataExplor()
    app.mainloop()