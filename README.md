💰 Fintech App Review Analysis

Project Overview

This project, conducted for Omega Consultancy, analyzes customer experience for three Ethiopian mobile banking applications—Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank—by scraping, processing, and analyzing user reviews from the Google Play Store.

The goal is to identify core satisfaction drivers and major pain points to guide product development and customer retention strategies.

🛠️ Tasks Completed (Interim Submission)

Task 1: Data Collection and Preprocessing (COMPLETED)

Source: Google Play Store (App IDs: com.combanketh.mobilebanking, com.boa.boaMobileBanking, com.dashen.dashensuperapp)

Data Quality: Successfully scraped 1,200+ clean reviews with 0% error rate.

Output: data/cleaned_reviews.csv

Task 2: Sentiment and Thematic Analysis (COMPLETED)

Sentiment Model: Hugging Face distilbert-base-uncased-finetuned-sst-2-english.

Thematic Analysis: Keyword extraction (TF-IDF) grouped into actionable themes (e.g., Transaction Performance, Reliability & Bugs).

Output: data/analyzed_reviews.csv and data/themes_summary.csv

⚙️ Project Structure

Fintech_App_Review_Analysis/
├── data/
│   ├── cleaned_reviews.csv      (Task 1 Output)
│   └── analyzed_reviews.csv     (Task 2 Output)
├── sql/
│   └── schema.sql               (Task 3 Schema)
├── src/
│   ├── T1_scrape_preprocess.py
│   ├── T2_sentiment_theme.py
│   └── T3_database_ingestion.py (Pending Execution)
├── .gitignore
├── README.md
└── requirements.txt


📦 Setup and Dependencies

This project relies on the Python packages listed in requirements.txt.

Installation Steps:

Clone the Repository:

git clone [YOUR GITHUB LINK HERE]
cd Fintech_App_Review_Analysis


Create/Activate Virtual Environment.

Install Python Libraries:

pip install -r requirements.txt


Install NLP Data (Crucial):

python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('all')"


🚧 Tasks Remaining (Final Submission)

Task 3: Store Cleaned Data in PostgreSQL

Status: Code Complete, Awaiting PostgreSQL Server Installation/Connection.

Configuration: Requires config.py with valid PostgreSQL credentials (DB_CONFIG).

Steps: Install PostgreSQL Server, Execute sql/schema.sql (via DataGrip/pgAdmin), Run T3_database_ingestion.py.

Task 4: Insights and Recommendations

Status: Pending.

Goal: Visualize data from PostgreSQL using Matplotlib/Seaborn and produce a final 10-page report.