"""
Training Script for Traditional ML Models
Trains Logistic Regression + SVM on clean data
OPTIMIZED FOR DESKTOP (Multi-core CPU) - 100K SAMPLES
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from datetime import datetime
import os


def load_data(file_path, sample_size=100000):
    """Load and preprocess data - LIMITED TO 100K SAMPLES"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    print(f"Total rows in dataset: {len(df)}")
    
    # Sample 100k rows randomly
    if len(df) > sample_size:
        print(f"Sampling {sample_size:,} rows randomly from {len(df):,} total rows...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"✓ Sampled {len(df):,} reviews")
    else:
        print(f"Dataset has {len(df)} rows (less than {sample_size:,}), using all rows")

    # Check columns
    print(f"Columns: {df.columns.tolist()}")

    # Clean and prepare
    df['text'] = df['text'].fillna('')
    df['label'] = (df['rating'] >= 4).astype(int)  # 4-5 stars = positive (1), 1-3 stars = negative (0)

    print(f"Final dataset size: {len(df):,} reviews")
    print(f"Positive reviews (4-5 stars): {(df['label'] == 1).sum():,}")
    print(f"Negative reviews (1-3 stars): {(df['label'] == 0).sum():,}")

    return df


def train_model(X_train, y_train, model_type='logistic'):
    """Train traditional ML model"""
    print(f"\n{'=' * 60}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'=' * 60}")

    # Vectorize text with optimized settings
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,  # Ignore rare terms
        max_df=0.95,  # Ignore very common terms
        strip_accents='unicode',
        lowercase=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)

    print(f"Training set size: {X_train_vec.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # Train model
    start_time = datetime.now()

    if model_type == 'logistic':
        print("Training Logistic Regression...")
        print("Using saga solver with L2 penalty (multi-core optimized)")
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            verbose=1,
            n_jobs=-1,  # Use all CPU cores
            solver='saga',  # Faster for large datasets
            penalty='l2'
        )
    else:
        print("Training SVM with linear kernel...")
        print("Using LinearSVC (optimized for linear kernel)")
        from sklearn.svm import LinearSVC
        model = LinearSVC(
            random_state=42,
            verbose=1,
            max_iter=1000,
            dual=False  # Faster when n_samples > n_features
        )

    model.fit(X_train_vec, y_train)

    training_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"Training complete in {training_time:.2f} minutes!")

    metrics = {
        'training_samples': len(X_train),
        'training_time_minutes': training_time
    }

    print(f"\nTraining Summary:")
    print(f"  Samples Trained: {metrics['training_samples']:,}")
    print(f"  Training Time: {training_time:.2f} minutes")

    return model, vectorizer, metrics


def save_model(model, vectorizer, metrics, model_name, output_dir='models'):
    """Save trained model, vectorizer, and metrics"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save model
    model_path = os.path.join(output_dir, f'{model_name}_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nSaved model to: {model_path}")

    # Save vectorizer
    vectorizer_path = os.path.join(output_dir, f'{model_name}_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Saved vectorizer to: {vectorizer_path}")

    # Save metrics
    metrics_data = {
        'model': model_name,
        'metrics': metrics,
        'timestamp': timestamp,
        'model_params': str(model.get_params())
    }
    metrics_path = os.path.join(output_dir, f'{model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")


def main():
    # Configuration - 100K SAMPLES FOR FAIRNESS
    DATA_PATH = 'data/books_500k_clean.csv'  # Will sample 100k from this
    SAMPLE_SIZE = 100000  # Use 100k samples for fairness

    print("=" * 60)
    print("TRADITIONAL ML MODEL TRAINING")
    print("OPTIMIZED FOR DESKTOP (Multi-core)")
    print("=" * 60)
    print(f"Dataset: {DATA_PATH}")
    print(f"Sample Size: {SAMPLE_SIZE:,} reviews (randomly sampled)")
    print(f"Training on ALL sampled rows (no test split)")
    print(f"Testing will be done on separate 10k dataset in Streamlit")
    print(f"Models: Logistic Regression, SVM")
    print("=" * 60)

    # Load data with sampling
    df = load_data(DATA_PATH, sample_size=SAMPLE_SIZE)

    print(f"\nUsing {len(df):,} reviews for training")
    X_train = df['text']
    y_train = df['label']

    # Train Logistic Regression
    lr_model, lr_vectorizer, lr_metrics = train_model(
        X_train, y_train, model_type='logistic'
    )
    save_model(lr_model, lr_vectorizer, lr_metrics, 'logistic_regression')

    # Train SVM
    svm_model, svm_vectorizer, svm_metrics = train_model(
        X_train, y_train, model_type='svm'
    )
    save_model(svm_model, svm_vectorizer, svm_metrics, 'svm')

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nModel Summary:")
    print(
        f"Logistic Regression - Trained on {lr_metrics['training_samples']:,} samples ({lr_metrics['training_time_minutes']:.1f} min)")
    print(
        f"SVM                 - Trained on {svm_metrics['training_samples']:,} samples ({svm_metrics['training_time_minutes']:.1f} min)")
    print("\nAll models saved to 'models/' directory")
    print("Next: Use the 10k dataset in Streamlit for testing!")
    print("=" * 60)


if __name__ == "__main__":
    main()