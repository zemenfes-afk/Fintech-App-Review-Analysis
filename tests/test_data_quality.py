# tests/test_data_quality.py

import unittest
import pandas as pd
import os

# Assuming data is in the data folder relative to the project root
CLEANED_REVIEWS_PATH = os.path.join('data', 'cleaned_reviews.csv')
MIN_EXPECTED_REVIEWS = 1200  # KPI minimum


class TestDataQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the cleaned dataset once for all tests."""
        if os.path.exists(CLEANED_REVIEWS_PATH):
            cls.df = pd.read_csv(CLEANED_REVIEWS_PATH)
        else:
            cls.df = pd.DataFrame()  # Empty if file not found

    def test_minimum_review_count(self):
        """Checks if the minimum KPI of 1200 reviews was met."""
        if self.df.empty:
            self.fail("Cleaned data file not found. Cannot run test.")
        self.assertGreaterEqual(len(self.df), MIN_EXPECTED_REVIEWS,
                                f"KPI failure: Collected only {len(self.df)} reviews, less than minimum {MIN_EXPECTED_REVIEWS}.")

    def test_no_missing_critical_data(self):
        """Checks for missing values in critical columns."""
        if self.df.empty:
            return
        # Critical columns that must be non-null
        critical_cols = ['review_id', 'review', 'rating', 'date', 'bank']
        for col in critical_cols:
            missing_count = self.df[col].isnull().sum()
            self.assertEqual(missing_count, 0,
                             f"Data quality error: {missing_count} missing values found in column '{col}'.")

    def test_rating_bounds(self):
        """Checks if all ratings are within the 1-5 star range."""
        if self.df.empty:
            return
        valid_ratings = self.df['rating'].between(1, 5, inclusive='both').all()
        self.assertTrue(valid_ratings, "Data quality error: Ratings found outside the 1 to 5 range.")


if __name__ == '__main__':
    unittest.main()