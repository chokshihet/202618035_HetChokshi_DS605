import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

raw_file_path = "data/books_raw.csv"

if not os.path.exists(raw_file_path):
    raise FileNotFoundError(
        f"The file '{raw_file_path}' was not found. "
        "Please run the Scrapy spider first: `scrapy crawl books -o data/books_raw.csv`"
    )

raw_df = pd.read_csv(raw_file_path)

print("TASK 1: RAW DATA REPORT\n\n")

print(f"Total Scraped Records: {len(raw_df)}")
print("\nMissing Values Count per Column:")
print(raw_df.isnull().sum())
print(f"\nDuplicate UPC Values Count: {raw_df['UPC'].duplicated().sum()}")
print("=" * 60 + "\n")

df = raw_df.copy()

df = df.drop_duplicates(subset=["UPC"], keep="first")

text_columns = ["title", "category", "product_description"]
for col in text_columns:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

df["product_description"] = df["product_description"].replace(
    ["nan", "None", "", "NaN"], "No description available"
)

df["price"] = df["price"].astype(str).str.extract(r"(\d+\.\d+)").astype(float)

rating_mapping = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
df["rating"] = df["rating"].map(rating_mapping).fillna(0).astype(int)

def extract_stock_count(val):
    match = re.search(r"(\d+)", str(val))
    return int(match.group(1)) if match else 0

df["available_stock_count"] = df["availability"].apply(extract_stock_count)

df["number_of_reviews"] = (
    pd.to_numeric(df["number_of_reviews"], errors="coerce")
    .fillna(0)
    .astype(int)
)

df["description_word_count"] = df["product_description"].apply(
    lambda text: len(text.split()) if text != "No description available" else 0
)

df["price_band"] = pd.qcut(
    df["price"], q=3, labels=["Budget", "Mid-Range", "Premium"]
)

df["value_score"] = (df["rating"] / df["price"]).round(4)

median_price = df["price"].median()
df["recommended"] = (df["rating"] >= 4) & (df["price"] < median_price)

cleaned_file_path = "data/books_cleaned.csv"
df.to_csv(cleaned_file_path, index=False)


print("TASK 2: CLEANED DATA SUMMARY\n\n")

print(f"Total Records After Deduplication: {len(df)}")
print(f"Cleaned dataset exported to: {cleaned_file_path}")
print("\nSummary Statistics for Numeric Fields:")
print(
    df[
        [
            "price",
            "rating",
            "available_stock_count",
            "description_word_count",
            "value_score",
        ]
    ].describe()
)
print("=" * 60 + "\n")

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(df["price"], kde=True, color="skyblue", bins=15)
plt.title("Plot 1: Book Price Distribution (£)", fontsize=14, fontweight="bold")
plt.xlabel("Price (£)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.savefig("plots/price_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="rating", hue="rating", palette="Blues_d", legend=False)
plt.title("Plot 2: Star Rating Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Star Rating (1-5)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.savefig("plots/rating_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
top_categories = (
    df.groupby("category")["price"].mean().sort_values(ascending=False).head(10)
)
sns.barplot(
    x=top_categories.values,
    y=top_categories.index,
    hue=top_categories.index,
    palette="viridis",
    legend=False
)
plt.title("Plot 3: Top 10 Most Expensive Categories (Average Price)", fontsize=14, fontweight="bold")
plt.xlabel("Average Price (£)", fontsize=12)
plt.ylabel("Category", fontsize=12)
plt.tight_layout()
plt.savefig("plots/avg_price_by_category.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="rating", y="price", hue="rating", palette="Set2", legend=False)
plt.title("Plot 4: Price Distribution Across Star Ratings", fontsize=14, fontweight="bold")
plt.xlabel("Star Rating", fontsize=12)
plt.ylabel("Price (£)", fontsize=12)
plt.tight_layout()
plt.savefig("plots/price_vs_rating.png", dpi=300)
plt.close()

valid_descriptions = df[df["product_description"] != "No description available"][
    "product_description"
]
combined_text = " ".join(valid_descriptions)

custom_stopwords = set(STOPWORDS).union(
    {"book", "story", "one", "will", "new", "life", "read", "novel"}
)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    stopwords=custom_stopwords,
    max_words=100,
    colormap="viridis"
).generate(combined_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud: Combined Book Descriptions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/wordcloud_descriptions.png", dpi=300)
plt.close()

print("=" * 60)
print("TASK 3 COMPLETE: Plots saved successfully in 'plots/' directory.")
print("Generated files:")
print("  - plots/price_distribution.png")
print("  - plots/rating_distribution.png")
print("  - plots/avg_price_by_category.png")
print("  - plots/price_vs_rating.png")
print("  - plots/wordcloud_descriptions.png\n")
