"""
Training Script for Modern ML/DL Models
Trains DistilBERT + SBERT on clean data
OPTIMIZED FOR RTX 3050 MOBILE (4GB VRAM) - 100K SAMPLES
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from datetime import datetime
import os


# Custom Dataset for DistilBERT
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


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


def train_distilbert(X_train, y_train, epochs=2, batch_size=16):
    """Train DistilBERT model - OPTIMIZED FOR RTX 3050 MOBILE (4GB VRAM)"""
    print(f"\n{'=' * 60}")
    print("Training DistilBERT Model (RTX 3050 Mobile Optimized)")
    print(f"{'=' * 60}")

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    print("Loading DistilBERT tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )

    print(f"Creating training dataset...")
    train_dataset = ReviewDataset(X_train.values, y_train.values, tokenizer)

    print(f"Training samples: {len(train_dataset):,}")

    # Training arguments - OPTIMIZED FOR 4GB VRAM
    training_args = TrainingArguments(
        output_dir='./distilbert_results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,  # Reduced to 16 for 4GB VRAM
        learning_rate=5e-5,
        warmup_steps=500,  # Adjusted for 100k dataset
        weight_decay=0.01,
        logging_dir='./distilbert_logs',
        logging_steps=200,  # Log every 200 steps
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,  # Enable mixed precision for NVIDIA GPUs (saves memory)
        dataloader_num_workers=2,  # Reduced for laptop
        gradient_accumulation_steps=2,  # Simulate batch_size=32
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size} (with gradient accumulation = effective batch size 32)")
    print("Estimated time: ~2-3 hours on RTX 3050 Mobile for 100k samples...")

    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()

    training_time = (end_time - start_time).total_seconds() / 60
    print(f"\nTraining completed in {training_time:.1f} minutes")

    metrics = {
        'training_samples': len(train_dataset),
        'training_time_minutes': training_time,
        'epochs': epochs,
        'batch_size': batch_size,
        'effective_batch_size': batch_size * 2  # Due to gradient accumulation
    }

    print(f"\nTraining Summary:")
    print(f"  Samples Trained: {metrics['training_samples']:,}")
    print(f"  Training Time: {training_time:.1f} minutes")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size} (effective: {metrics['effective_batch_size']})")

    return model, tokenizer, metrics


def train_sbert(X_train, y_train, batch_size=32):
    """Train SBERT model - OPTIMIZED FOR RTX 3050 MOBILE (4GB VRAM)"""
    print(f"\n{'=' * 60}")
    print("Training SBERT Model (RTX 3050 Mobile Optimized)")
    print(f"{'=' * 60}")

    print("Loading SBERT model (all-MiniLM-L6-v2)...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Move to GPU if available
    if torch.cuda.is_available():
        sbert_model = sbert_model.cuda()
        print("SBERT model moved to GPU")

    print(f"\nEncoding training data ({len(X_train):,} samples)...")
    print(f"Batch size: {batch_size} (safe for 4GB VRAM)")

    start_time = datetime.now()
    train_embeddings = sbert_model.encode(
        X_train.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    encoding_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"Encoding completed in {encoding_time:.1f} minutes")

    print(f"\nTraining Logistic Regression classifier on embeddings...")
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        verbose=1,
        n_jobs=-1  # Use all CPU cores
    )
    classifier.fit(train_embeddings, y_train)

    total_time = (datetime.now() - start_time).total_seconds() / 60

    metrics = {
        'training_samples': len(X_train),
        'training_time_minutes': total_time,
        'batch_size': batch_size
    }

    print(f"\nTraining Summary:")
    print(f"  Samples Trained: {metrics['training_samples']:,}")
    print(f"  Total Time: {total_time:.1f} minutes")
    print(f"  Batch Size: {batch_size}")

    return sbert_model, classifier, metrics


def save_distilbert(model, tokenizer, metrics, output_dir='models'):
    """Save DistilBERT model"""
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, 'distilbert')
    os.makedirs(model_dir, exist_ok=True)

    print(f"\nSaving DistilBERT model to: {model_dir}")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Save metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_data = {
        'model': 'distilbert',
        'metrics': metrics,
        'timestamp': timestamp,
        'gpu': 'RTX 3050 Mobile (4GB)',
        'batch_size': metrics['batch_size'],
        'effective_batch_size': metrics['effective_batch_size']
    }
    metrics_path = os.path.join(output_dir, 'distilbert_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")


def save_sbert(sbert_model, classifier, metrics, output_dir='models'):
    """Save SBERT model"""
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, 'sbert')
    os.makedirs(model_dir, exist_ok=True)

    print(f"\nSaving SBERT model to: {model_dir}")
    sbert_model.save(model_dir)

    # Save classifier
    classifier_path = os.path.join(output_dir, 'sbert_classifier.pkl')
    with open(classifier_path, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Saved classifier to: {classifier_path}")

    # Save metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_data = {
        'model': 'sbert',
        'metrics': metrics,
        'timestamp': timestamp,
        'gpu': 'RTX 3050 Mobile (4GB)',
        'batch_size': metrics['batch_size']
    }
    metrics_path = os.path.join(output_dir, 'sbert_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")


def main():
    # Configuration - OPTIMIZED FOR RTX 3050 MOBILE (4GB VRAM) + 100K SAMPLES
    DATA_PATH = 'data/books_500k_clean.csv'  # Will sample 100k from this
    SAMPLE_SIZE = 100000  # Use 100k samples for fairness

    DISTILBERT_EPOCHS = 2  # Reduced for faster testing
    DISTILBERT_BATCH_SIZE = 16  # Safe for 4GB VRAM
    SBERT_BATCH_SIZE = 32  # Safe for 4GB VRAM

    print("=" * 60)
    print("MODERN ML/DL MODEL TRAINING")
    print("OPTIMIZED FOR RTX 3050 MOBILE (4GB VRAM)")
    print("=" * 60)
    print(f"Dataset: {DATA_PATH}")
    print(f"Sample Size: {SAMPLE_SIZE:,} reviews (randomly sampled)")
    print(f"Training on ALL sampled rows (no test split)")
    print(f"Testing will be done on separate dataset in Streamlit")
    print(f"Models: DistilBERT, SBERT")
    print(f"DistilBERT batch size: {DISTILBERT_BATCH_SIZE} (effective: 32 with gradient accumulation)")
    print(f"SBERT batch size: {SBERT_BATCH_SIZE}")
    print(f"Mixed precision (fp16): Enabled for NVIDIA GPU")
    print("=" * 60)

    # GPU check
    if torch.cuda.is_available():
        print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("\n⚠ WARNING: No GPU detected! Training will be very slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Load data with sampling
    df = load_data(DATA_PATH, sample_size=SAMPLE_SIZE)

    print(f"\nUsing {len(df):,} reviews for training")
    X_train = df['text']
    y_train = df['label']

    # Train DistilBERT
    distilbert_model, distilbert_tokenizer, distilbert_metrics = train_distilbert(
        X_train, y_train,
        epochs=DISTILBERT_EPOCHS,
        batch_size=DISTILBERT_BATCH_SIZE
    )
    save_distilbert(distilbert_model, distilbert_tokenizer, distilbert_metrics)

    # Train SBERT
    sbert_model, sbert_classifier, sbert_metrics = train_sbert(
        X_train, y_train,
        batch_size=SBERT_BATCH_SIZE
    )
    save_sbert(sbert_model, sbert_classifier, sbert_metrics)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nModel Summary:")
    print(f"DistilBERT - Trained on {distilbert_metrics['training_samples']:,} samples in {distilbert_metrics['training_time_minutes']:.1f} min")
    print(f"SBERT      - Trained on {sbert_metrics['training_samples']:,} samples in {sbert_metrics['training_time_minutes']:.1f} min")
    print("\nAll models saved to 'models/' directory")
    print("Next: Use these models in Streamlit for testing!")
    print("\n📦 Models ready for demo!")
    print("=" * 60)


if __name__ == "__main__":
    main()