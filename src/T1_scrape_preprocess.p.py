# src/T1_scrape_preprocess.py

import pandas as pd
from google_play_scraper import Sort, reviews
import numpy as np
from datetime import datetime
import os

# --- Configuration ---
# *** These are the CORRECTED App IDs based on your links ***
APP_IDS = {
    "Commercial Bank of Ethiopia": "com.combanketh.mobilebanking",
    "Bank of Abyssinia": "com.boa.boaMobileBanking",
    "Dashen Bank": "com.dashen.dashensuperapp"
}
MIN_REVIEWS_PER_BANK = 400
OUTPUT_DIR = 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define expected columns once for consistent empty DataFrame creation
EXPECTED_COLUMNS = ['review_text', 'rating', 'date', 'source', 'bank']


def scrape_reviews(bank_name, app_id, count=MIN_REVIEWS_PER_BANK):
    """
    Scrape reviews for a given Google Play app ID.
    Handles potential scraping errors and ensures consistent DataFrame columns.
    """

    print(f"Scraping reviews for {bank_name} (ID: {app_id})...")

    try:
        result, continuation_token = reviews(
            app_id,
            lang='en',  # Filter for English reviews for consistent NLP
            country='us',  # Using a default country, but reviews are global
            sort=Sort.NEWEST,
            count=count,
            filter_score_with=None
        )

        # The 'content' field holds the review text
        scraped_data = [{
            'review_text': r['content'],
            'rating': r['score'],
            'date': r['at'],
            'source': 'Google Play',
            'bank': bank_name
        } for r in result]

        if scraped_data:
            print(f"Successfully scraped {len(scraped_data)} reviews for {bank_name}.")
            return pd.DataFrame(scraped_data)
        else:
            print(f"Warning: Scraper returned 0 reviews for {bank_name}. Returning empty DataFrame.")
            # Return an empty DataFrame with the correct column names
            return pd.DataFrame(columns=EXPECTED_COLUMNS)

    except Exception as e:
        print(f"CRITICAL ERROR during scraping {bank_name} ({app_id}): {e}")
        # Return an empty DataFrame on error to prevent downstream KeyError
        return pd.DataFrame(columns=EXPECTED_COLUMNS)


def preprocess_data(df):
    """Clean and preprocess the raw DataFrame."""
    print("Starting preprocessing...")

    # Check if the DataFrame is empty before proceeding
    if df.empty:
        print("Preprocessing skipped: DataFrame is empty.")
        # Return an empty DataFrame with final expected columns
        return pd.DataFrame(columns=['review_id', 'review', 'rating', 'date', 'bank', 'source'])

    # 1. Handle Duplicates and Missing Data
    required_cols = ['review_text', 'bank', 'rating', 'date']

    df.drop_duplicates(subset=required_cols, inplace=True)

    # Remove rows where review_text is missing or empty
    df.dropna(subset=['review_text'], inplace=True)
    df = df[df['review_text'].astype(str).str.strip() != ""]

    # 2. Normalize Dates
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.normalize().dt.strftime('%Y-%m-%d')

    # 3. Rename columns for clarity and schema adherence
    df.rename(columns={'review_text': 'review'}, inplace=True)

    # 4. Add a unique ID for the reviews table
    df.reset_index(drop=True, inplace=True)
    df.index.name = 'review_id'
    df = df.reset_index()

    print(f"Preprocessing complete. Total clean reviews: {len(df)}")
    return df


if __name__ == "__main__":
    all_reviews_list = []

    for bank_name, app_id in APP_IDS.items():
        # Pass both bank_name and app_id to the function
        df_bank = scrape_reviews(bank_name, app_id, count=MIN_REVIEWS_PER_BANK)
        all_reviews_list.append(df_bank)

    # Combine all bank data
    raw_df = pd.concat(all_reviews_list, ignore_index=True)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, 'raw_reviews.csv'), index=False)
    print(f"Saved raw data to {os.path.join(OUTPUT_DIR, 'raw_reviews.csv')}. Total raw: {len(raw_df)}")

    # Preprocess
    cleaned_df = preprocess_data(raw_df)

    # Save cleaned data
    cleaned_df.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_reviews.csv'), index=False)
    print(f"Saved cleaned data to {os.path.join(OUTPUT_DIR, 'cleaned_reviews.csv')}.")

    # Data Quality Check (KPI)
    total_reviews = len(cleaned_df)
    total_expected = len(APP_IDS) * MIN_REVIEWS_PER_BANK
    print(f"\n--- Data Quality Summary ---")
    print(f"Target Reviews: {total_expected}")
    print(f"Collected Clean Reviews: {total_reviews}")
    # Handle division by zero if raw_df is empty
    raw_len = len(raw_df) if len(raw_df) > 0 else 1
    print(f"Missing/Duplicate/Empty Rate: {1 - (total_reviews / raw_len):.2%}")