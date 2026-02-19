import pandas as pd
from datasets import load_dataset
import os

# 1. Data Loading
print("🔄 Fetching dataset from Hugging Face...")
# We load the first 50 reviews to create our "Gold Standard" training set
dataset = load_dataset("yelp_review_full", split="train[:50]", trust_remote_code=True)

texts = dataset['text']
original_ratings = dataset['label'] # Original 1-5 star ratings
labels = []
output_file = "yelp_train_dataset.csv"


# 2. Interactive Labeling Loop
for i, text in enumerate(texts):
    print(f"\n[Record {i + 1} of 50]")
    print(f"Review Content: {text[:300]}...") # Showing first 300 chars

    # Professional Assignment Prompt
    category = input("\n Assign a category for this review: ").strip()

    # Quit mechanism
    if category.lower() == 'q':
        print("\nSaving progress and terminating...")
        break

    labels.append(category)

    # 3. DYNAMIC COLUMN ADDITION WITH PANDAS
    # We create/update the DataFrame at each step to ensure data safety
    current_df = pd.DataFrame({
        'label': [r + 1 for r in original_ratings[:len(labels)]], # Orijinal label (1-5 arası),
        'Sample_50': [t[:60].replace("\n", " ").strip() + "..." for t in texts[:len(labels)]],
        'Categories': labels,              # This is the 3rd column I manually added

    })

    # 4. SAVE TO CSV & LIVE PREVIEW
    # We use utf-8-sig for Excel compatibility
    current_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\n📊 CURRENT PROGRESS TABLE (Saved to CSV):")
    print("-" * 70)
    print(current_df.to_string(index=False))
    print("-" * 70)

print(f"\n✅ Task completed successfully!")
print(f"Your training dataset is ready at: '{os.path.abspath(output_file)}'")