import pandas as pd
import psycopg2
import os
from config import DB_CONFIG

# --- Configuration ---
INPUT_FILE = os.path.join('data', 'analyzed_reviews.csv')


def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Database connection successful.")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to the database. Check config.py and ensure the PostgreSQL service is running: {e}")
        return None


def insert_reviews_data(conn, df):
    """Inserts processed reviews into the reviews table."""
    cursor = conn.cursor()

    # 1. Fetch bank IDs - CRITICAL FIX: Use 'public.banks'
    bank_ids = {}
    cursor.execute("SELECT bank_name, bank_id FROM public.banks;")
    for name, id in cursor.fetchall():
        bank_ids[name] = id

    if not bank_ids:
        print("Error: Banks table is empty. Please run schema.sql first.")
        return

    # 2. Prepare data for insertion
    df['bank_id'] = df['bank'].map(bank_ids).astype('Int64')

    # Filter out reviews where the bank name didn't match an ID
    df_to_insert = df.dropna(subset=['bank_id'])

    # SQL INSERT statement - CRITICAL FIX: Use 'public.reviews'
    insert_query = """
    INSERT INTO public.reviews (
        review_id, bank_id, review_text, rating, review_date, 
        sentiment_label, sentiment_score, normalized_sentiment, final_sentiment, identified_theme, source
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (review_id) DO NOTHING;
    """

    # 3. Execute insertion
    count = 0
    for index, row in df_to_insert.iterrows():
        try:
            cursor.execute(insert_query, (
                row['review_id'],
                row['bank_id'],
                row['review'],
                int(row['rating']),
                row['date'],
                row['sentiment_label'],
                row['sentiment_score'],
                row['normalized_sentiment'],
                row['final_sentiment'],
                row['identified_theme'],
                row['source']
            ))
            count += 1
        except Exception as e:
            # Print the error but don't stop the whole process immediately
            print(f"Error inserting row {row['review_id']}: {e}")
            conn.rollback()

    conn.commit()
    print(f"Successfully inserted {count} reviews into the database.")
    cursor.close()


def verify_data_integrity(conn):
    """Executes SQL queries to verify the data."""
    cursor = conn.cursor()
    print("\n--- Data Verification Queries ---")

    # Query 1: Count reviews per bank - CRITICAL FIX: Use 'public.banks'/'public.reviews'
    cursor.execute("""
        SELECT b.bank_name, COUNT(r.review_id) AS review_count
        FROM public.reviews r
        JOIN public.banks b ON r.bank_id = b.bank_id
        GROUP BY b.bank_name;
    """)
    print("Reviews per Bank:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} reviews")

    # Query 2: Average rating per bank - CRITICAL FIX: Use 'public.banks'/'public.reviews'
    cursor.execute("""
        SELECT b.bank_name, ROUND(AVG(r.rating)::numeric, 2) AS avg_rating
        FROM public.reviews r
        JOIN public.banks b ON r.bank_id = b.bank_id
        GROUP BY b.bank_name;
    """)
    print("Average Rating per Bank:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    cursor.close()


if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Run T2_sentiment_theme.py first.")
    else:
        df = pd.read_csv(INPUT_FILE)
        conn = get_db_connection()

        if conn:
            insert_reviews_data(conn, df)
            verify_data_integrity(conn)
            conn.close()