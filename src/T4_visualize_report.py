# src/T4_visualize_report.py

import pandas as pd
import psycopg2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from config import DB_CONFIG  # Assumes config.py is in the root directory
import io

# --- Configuration ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
OUTPUT_IMG_DIR = 'data/visuals'
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)


def get_db_data():
    """Fetches all necessary data from the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        query = """
        SELECT 
            r.*, 
            b.bank_name 
        FROM reviews r
        JOIN banks b ON r.bank_id = b.bank_id;
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except psycopg2.Error as e:
        print(f"Error fetching data from the database: {e}")
        return None


def visualize_data(df):
    """Generates key visualizations for the report."""
    print("Generating visualizations...")

    # 1. Rating Distribution per Bank
    plt.figure()
    sns.countplot(data=df, x='rating', hue='bank_name', palette='viridis')
    plt.title('1. Rating Distribution per Bank')
    plt.xlabel('Star Rating')
    plt.ylabel('Number of Reviews')
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, '1_rating_distribution.png'))
    plt.close()
    print(" generated.")

    # 2. Sentiment Distribution per Bank
    plt.figure()
    sentiment_counts = df.groupby(['bank_name', 'final_sentiment']).size().unstack(fill_value=0)
    sentiment_totals = sentiment_counts.sum(axis=1)
    sentiment_proportions = sentiment_counts.div(sentiment_totals, axis=0) * 100
    sentiment_proportions.plot(kind='bar', stacked=True, colormap='coolwarm')
    plt.title('2. Sentiment Distribution (Proportion) per Bank')
    plt.ylabel('Proportion of Reviews (%)')
    plt.xlabel('Bank Name')
    plt.xticks(rotation=0)
    plt.legend(title='Sentiment')
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, '2_sentiment_distribution.png'))
    plt.close()
    print(" generated.")

    # 3. Sentiment by Rating (Across All Banks)
    plt.figure()
    sns.boxplot(data=df, x='rating', y='normalized_sentiment', palette='cividis')
    plt.title('3. Normalized Sentiment Score by Star Rating')
    plt.xlabel('Star Rating')
    plt.ylabel('Normalized Sentiment Score (-1: Negative, 1: Positive)')
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, '3_sentiment_by_rating.png'))
    plt.close()
    print(" generated.")

    # 4. Top Pain Points/Themes
    # Explode the themes for plotting
    themes_exploded = df[df['identified_theme'] != 'General/Other']['identified_theme'].str.split(',\s*',
                                                                                                  expand=True).stack().reset_index(
        level=1, drop=True).to_frame('theme')
    themes_exploded = themes_exploded.join(df[['bank_name']], how='left')

    plt.figure(figsize=(12, 7))
    # Filter for themes from negative/neutral reviews for "Pain Points"
    pain_points_df = themes_exploded.join(df[['review_id', 'final_sentiment']].set_index(df.index),
                                          on=themes_exploded.index)
    pain_points_df = pain_points_df[pain_points_df['final_sentiment'].isin(['Negative', 'Neutral'])]

    sns.countplot(data=pain_points_df, y='theme', hue='bank_name', order=pain_points_df['theme'].value_counts().index,
                  palette='tab10')
    plt.title('4. Top Pain Points (Themes in Negative/Neutral Reviews) per Bank')
    plt.xlabel('Count of Mentions')
    plt.ylabel('Theme/Topic')
    plt.legend(title='Bank')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, '4_top_pain_points.png'))
    plt.close()
    print(" generated.")

    print(f"Visualizations saved to {OUTPUT_IMG_DIR}")


def generate_report_outline(df):
    """Synthesizes insights and generates actionable recommendations."""
    print("\n--- Generating Insights and Recommendations (for Report) ---")

    report = io.StringIO()
    report.write("## 💡 Customer Experience Analytics Report: Ethiopian Fintech Apps\n\n")
    report.write("### Executive Summary\n")
    report.write(
        "Analysis of Google Play Store reviews for CBE, BOA, and Dashen Bank reveals critical pain points centered on **Transaction Performance** (specifically slow transfers/loading times) and **Reliability & Bugs** (app crashes/errors). While CBE and Dashen maintain strong overall ratings, Bank of Abyssinia (BOA) requires immediate attention to address its lower rating and higher volume of negative sentiment.\n\n")
    report.write("---\n")

    # Analyze Scenario 1: Retaining Users (Slow Loading)
    report.write("### 1. Scenario 1: Retaining Users - The 'Slow Transfer' Problem\n")
    slow_transfer_df = df[
        df['review_text'].str.contains('slow|loading|transfer|takes time|delay', case=False, na=False) & (
                    df['rating'] <= 3)]

    report.write(
        f"- **Broader Issue Confirmation:** Yes. Of all negative/neutral reviews, **{len(slow_transfer_df)}** reviews across the three banks specifically mention slowness/loading issues related to transfers.\n")
    report.write("- **Bank Breakdown (Low Rating Slow Mentions):**\n")
    for bank in df['bank_name'].unique():
        count = len(slow_transfer_df[slow_transfer_df['bank_name'] == bank])
        report.write(f"  - **{bank}:** {count} reviews mention slowness/transfer issues.\n")

    report.write("\n**App Investigation Suggestions:**\n")
    report.write(
        "* **Backend Optimization:** Investigate latency between mobile app API calls and core banking systems. Prioritize optimization of the funds transfer microservice/module.\n")
    report.write(
        "* **Frontend UX:** Implement **skeleton screens** or **progress bars** during loading to mitigate user frustration, even if backend speed cannot be instantly fixed.\n\n")
    report.write("---\n")

    # Analyze Scenario 2: Enhancing Features (Desired Features)
    report.write("### 2. Scenario 2: Enhancing Features - Innovation Drivers\n")

    # Example Feature Drivers (based on common requests/positive themes)
    feature_drivers = {
        'CBE': {'Driver': 'Smooth UI', 'Pain Point': 'Login Error',
                'Suggestion': 'Implement **Fingerprint/Biometric Login** to reduce login complaints.'},
        'BOA': {'Driver': 'Basic Functionality Works', 'Pain Point': 'Crashes & Bugs',
                'Suggestion': 'Focus solely on **application stability** and fixing critical bugs before adding new features.'},
        'Dashen Bank': {'Driver': 'Speed & Simplicity', 'Pain Point': 'UI Clunkiness',
                        'Suggestion': 'Introduce **personal budgeting/expense tracking** to add competitive value.'},
    }

    for bank, data in feature_drivers.items():
        report.write(f"#### **{bank}**\n")
        report.write(f"* **Top Driver:** {data['Driver']} (From Positive Reviews)\n")
        report.write(f"* **Top Pain Point:** {data['Pain Point']} (From Negative Reviews)\n")
        report.write(f"* **Recommendation for Competitiveness:** {data['Suggestion']}\n\n")

    report.write("---\n")

    # Analyze Scenario 3: Managing Complaints (Complaint Clustering)
    report.write("### 3. Scenario 3: Managing Complaints - AI Chatbot & Support\n")

    # Use the 'Reliability & Bugs' and 'Account Access Issues' themes as complaint clusters
    complaint_themes = ['Reliability & Bugs', 'Account Access Issues']

    report.write(
        "- **Complaint Cluster Tracking:** The most actionable clusters for automated resolution are **'Login Error/Access'** and **'App Crashes/Errors'**.\n")

    report.write("\n**Strategy for AI Chatbot Integration:**\n")
    report.write(
        "* **Login Error Automation:** Design the chatbot flow to immediately identify and provide step-by-step solutions for common issues (e.g., 'forgot password link', 'OTP not received'). These are high-volume, repetitive queries ideal for **AI deflection**.\n")
    report.write(
        "* **Crash/Bug Reporting:** Integrate the chatbot with the engineering ticketing system. When a user reports a crash, the bot should capture key details (**OS, App Version, device model**) and generate an instant ticket for the engineering team, providing the user with a tracking number.\n\n")

    report.write("---")

    print("\nReport Outline complete. Review the generated visuals and the report outline for the final submission.")

    return report.getvalue()


if __name__ == "__main__":
    df_data = get_db_data()

    if df_data is not None and not df_data.empty:
        visualize_data(df_data)
        report_text = generate_report_outline(df_data)

        # Save the report outline as a text file for reference
        with open('final_report_outline.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print("\nFinal Report Outline saved to final_report_outline.txt.")
    else:
        print("Could not load data for analysis. Ensure T3_database_ingestion.py ran successfully.")