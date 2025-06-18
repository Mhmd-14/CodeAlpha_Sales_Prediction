import os
import sys
sys.path.append(os.path.abspath("C:/Users/ADMIN/Desktop/Data Science/Code Alpha Internship/Task_4_Sales_Prediction/"))
import pandas as pd
import tempfile
from src.preprocess import preprocess_data

def test_preprocess_data():
    # Create a temporary raw CSV file
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_path = os.path.join(tmp_dir, 'raw.csv')
        processed_dir = os.path.join(tmp_dir, 'processed')

        # Sample data with an index column to drop
        df = pd.DataFrame({
            'Unnamed: 0': [0, 1],
            'TV': [230.1, 44.5],
            'Radio': [37.8, 39.3],
            'Newspaper': [69.2, 41.3],
            'Sales': [22.1, 10.4]
        })
        df.to_csv(raw_path, index=False)

        # Run preprocessing
        output_path = preprocess_data(raw_path, processed_dir)

        # Check file exists
        assert os.path.exists(output_path)

        # Check content
        processed_df = pd.read_csv(output_path)
        assert processed_df.shape == (2, 4)  # First column should be dropped
        assert 'Unnamed: 0' not in processed_df.columns
