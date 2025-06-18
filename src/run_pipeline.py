from src.preprocess import preprocess_data
from src.train_evaluate import train_model

def run_pipeline(raw_data_path: str):
    print("🚀 [PIPELINE] Starting preprocessing...")
    processed_data_path = preprocess_data(raw_data_path, output_dir='../data/processed')

    print("🤖 [PIPELINE] Starting training...")
    train_model(processed_data_path, model_path='../models/model.pkl')

    print("✅ [PIPELINE] Pipeline completed successfully.")