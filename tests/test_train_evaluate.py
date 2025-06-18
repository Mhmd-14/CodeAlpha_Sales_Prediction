import os
import sys
sys.path.append(os.path.abspath("C:/Users/ADMIN/Desktop/Data Science/Code Alpha Internship/Task_4_Sales_Prediction/"))
import pandas as pd
import pytest
import tempfile
import joblib
from src.train_evaluate import train_model


def test_train_model():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create sample processed data
        data_path = os.path.join(tmp_dir, 'processed.csv')
        model_path = os.path.join(tmp_dir, 'model.pkl')

        df = pd.DataFrame({
            'TV': [230.1, 44.5, 17.2],
            'Radio': [37.8, 39.3, 45.9],
            'Newspaper': [69.2, 41.3, 69.3],
            'Sales': [22.1, 10.4, 9.3]
        })
        df.to_csv(data_path, index=False)

        # Train and save model
        train_model(data_path, model_path)

        # Check model file saved
        assert os.path.exists(model_path)

        # Load and verify it is a scikit-learn model
        model = joblib.load(model_path)
        assert hasattr(model, 'predict')