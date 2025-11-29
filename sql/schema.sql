-- sql/schema.sql

-- Drop tables if they exist to allow clean redeployment
DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS banks;

-- 1. Banks Table
CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) UNIQUE NOT NULL,
    app_name VARCHAR(255) UNIQUE NOT NULL
);

-- 2. Reviews Table
CREATE TABLE reviews (
    review_id INTEGER PRIMARY KEY, -- Use the 'review_id' from the DataFrame index
    bank_id INTEGER REFERENCES banks(bank_id), -- Foreign Key to Banks table
    review_text TEXT NOT NULL,
    rating INTEGER NOT NULL,
    review_date DATE NOT NULL,
    sentiment_label VARCHAR(50), -- e.g., 'Positive', 'Negative', 'Neutral'
    sentiment_score NUMERIC(5, 4), -- Original score from model
    normalized_sentiment NUMERIC(5, 4), -- Simplified score (-1 to 1)
    final_sentiment VARCHAR(50), -- 'Positive', 'Negative', 'Neutral'
    identified_theme TEXT, -- Comma-separated list of themes
    source VARCHAR(50) NOT NULL
);

-- Optional: Create indices for faster lookups
CREATE INDEX idx_reviews_bank_id ON reviews (bank_id);
CREATE INDEX idx_reviews_rating ON reviews (rating);
CREATE INDEX idx_reviews_date ON reviews (review_date);

-- Insert initial bank data (App names are based on the final App IDs you used)
INSERT INTO banks (bank_name, app_name) VALUES
('Commercial Bank of Ethiopia', 'com.combanketh.mobilebanking'),
('Bank of Abyssinia', 'com.boa.boaMobileBanking'),
('Dashen Bank', 'com.dashen.dashensuperapp');
```
eof

#### Step 2: Configure Database Credentials

Ensure your **`config.py`** file (in your project root) has the correct connection details for your local PostgreSQL installation:

```python
# config.py

# PostgreSQL Database Credentials
DB_CONFIG = {
    "dbname": "bank_reviews", # Make sure you created a DB with this name
    "user": "your_postgres_user", # <<< UPDATE THIS >>>
    "password": "your_postgres_password", # <<< UPDATE THIS >>>
    "host": "localhost",
    "port": "5432"
}
```

#### Step 3: Run the Ingestion Script

Now, run the Python script to connect to the DB and insert the data from `data/analyzed_reviews.csv`.

```bash
python C:\Users\HP\PycharmProjects\Fintech_App_Review_Analysis\src\T3_database_ingestion.py