import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import time
import os
from datetime import datetime
from redis import Redis, ConnectionPool
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from itertools import combinations


# Redis connection
def connect_redis(max_retries=5):
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    pool = ConnectionPool(
        host=redis_host,
        port=6379,
        db=0,
        decode_responses=True,
        max_connections=50,
        socket_keepalive=True,
        socket_connect_timeout=5,
        retry_on_timeout=True
    )

    for i in range(max_retries):
        try:
            client = Redis(connection_pool=pool)
            client.ping()
            return client
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(2)
            else:
                raise


redis_client = connect_redis()


# Load pre-trained models
@st.cache_resource
def load_pretrained_models():
    """Load all pre-trained models from disk"""
    models = {}

    try:
        # Load Logistic Regression
        if os.path.exists('models/logistic_regression_model.pkl'):
            with open('models/logistic_regression_model.pkl', 'rb') as f:
                lr_model = pickle.load(f)
            with open('models/logistic_regression_vectorizer.pkl', 'rb') as f:
                lr_vectorizer = pickle.load(f)
            models['Logistic Regression'] = {'model': lr_model, 'vectorizer': lr_vectorizer}
            st.sidebar.success("Logistic Regression loaded")

        # Load SVM
        if os.path.exists('models/svm_model.pkl'):
            with open('models/svm_model.pkl', 'rb') as f:
                svm_model = pickle.load(f)
            with open('models/svm_vectorizer.pkl', 'rb') as f:
                svm_vectorizer = pickle.load(f)
            models['SVM'] = {'model': svm_model, 'vectorizer': svm_vectorizer}
            st.sidebar.success("SVM loaded")

        # Load SBERT
        if os.path.exists('models/sbert'):
            sbert_model = SentenceTransformer('models/sbert')
            with open('models/sbert_classifier.pkl', 'rb') as f:
                sbert_classifier = pickle.load(f)
            models['SBERT'] = {'model': sbert_model, 'classifier': sbert_classifier}
            st.sidebar.success("SBERT loaded")

        # Load DistilBERT
        if os.path.exists('models/distilbert'):
            tokenizer = DistilBertTokenizer.from_pretrained('models/distilbert')
            model = DistilBertForSequenceClassification.from_pretrained('models/distilbert')
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            models['DistilBERT'] = {'model': model, 'tokenizer': tokenizer, 'device': device}
            st.sidebar.success("DistilBERT loaded")

    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")

    return models


def predict_with_model(model_info, texts, model_name):
    """Make predictions using pre-trained model"""
    if model_name in ['Logistic Regression', 'SVM']:
        vectorizer = model_info['vectorizer']
        model = model_info['model']
        X = vectorizer.transform(texts)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

    elif model_name == 'SBERT':
        sbert_model = model_info['model']
        classifier = model_info['classifier']
        embeddings = sbert_model.encode(texts.tolist(), show_progress_bar=False)
        predictions = classifier.predict(embeddings)
        probabilities = classifier.predict_proba(embeddings)

    elif model_name == 'DistilBERT':
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        device = model_info['device']

        predictions = []
        probabilities = []

        for text in texts:
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            predictions.append(pred)
            probabilities.append(probs.cpu().numpy()[0])

        predictions = np.array(predictions)
        probabilities = np.array(probabilities)

    return predictions, probabilities


def predict_with_ensemble(model_infos, texts, ensemble_type='voting'):
    """Make predictions using ensemble of models"""
    all_predictions = []
    all_probabilities = []

    for model_name, model_info in model_infos.items():
        preds, probs = predict_with_model(model_info, texts, model_name)
        all_predictions.append(preds)
        if probs is not None:
            all_probabilities.append(probs)

    # Convert to arrays
    all_predictions = np.array(all_predictions)

    if ensemble_type == 'voting':
        # Majority voting
        predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=all_predictions
        )

        # Average probabilities if available
        if all_probabilities:
            probabilities = np.mean(all_probabilities, axis=0)
        else:
            probabilities = None

    elif ensemble_type == 'averaging':
        # Probability averaging (requires all models to have probabilities)
        if all_probabilities:
            probabilities = np.mean(all_probabilities, axis=0)
            predictions = np.argmax(probabilities, axis=1)
        else:
            # Fallback to voting
            predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=all_predictions
            )
            probabilities = None

    return predictions, probabilities


def load_data_from_file(uploaded_file, sample_size=None):
    """Load data from uploaded file without caching"""
    uploaded_file.seek(0)  # Reset file pointer
    df = pd.read_csv(uploaded_file)

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    df['text'] = df['text'].fillna('')
    df['label'] = (df['rating'] >= 4).astype(int)

    return df


def distribute_reviews_to_users(df):
    """Distribute reviews to Docker user containers via Redis"""
    redis_client.delete('review_queue')
    redis_client.delete('feedback_queue')

    for idx, row in df.iterrows():
        review_data = {
            'id': idx,
            'text': row['text'],
            'rating': int(row['rating']),
            'timestamp': datetime.now().isoformat()
        }
        redis_client.rpush('review_queue', json.dumps(review_data))

    return redis_client.llen('review_queue')


def collect_feedback_from_users(expected_count, timeout=300):
    """Collect processed feedback with real-time progress"""
    collected = []
    progress_bar = st.progress(0)
    status_col1, status_col2, status_col3 = st.columns(3)

    start_time = time.time()
    last_update = 0

    while len(collected) < expected_count:
        if time.time() - start_time > timeout:
            st.warning(f"Timeout reached. Collected {len(collected)}/{expected_count}")
            break

        feedback = redis_client.lpop('feedback_queue')
        if feedback:
            collected.append(json.loads(feedback))

            # Update progress every 100ms
            if time.time() - last_update > 0.1:
                progress = len(collected) / expected_count
                progress_bar.progress(progress)

                poisoned = sum(1 for f in collected if f['is_poisoned'])
                legitimate = len(collected) - poisoned

                status_col1.metric("Total Processed", len(collected))
                status_col2.metric("Poisoned", poisoned)
                status_col3.metric("Legitimate", legitimate)

                last_update = time.time()
        else:
            time.sleep(0.05)

    progress_bar.empty()

    return pd.DataFrame(collected)


# Streamlit UI
st.set_page_config(page_title="Data Poisoning Simulation", layout="wide")

st.markdown("""
    <style>
        /* Set initial sidebar width */
        [data-testid="stSidebar"] {
            min-width: 350px;
            max-width: 350px;
        }

        /* Allow resizing */
        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 350px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Data Poisoning Attack Simulation")

# Sidebar - Simplified
st.sidebar.header("Configuration")

# Load models
st.sidebar.subheader("Pre-trained Models")
loaded_models = load_pretrained_models()

if not loaded_models:
    st.sidebar.warning(
        "No pre-trained models found. Please train models first using `train_traditional_models.py` and `train_modern_models.py`")

# Dataset upload
st.sidebar.subheader("Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file:
    # Quick preview to show file is loaded
    uploaded_file.seek(0)
    temp_df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"{len(temp_df):,} reviews loaded")

# Sample size
use_sampling = st.sidebar.checkbox("Use subset", value=True)
sample_size = None
if use_sampling:
    sample_size = st.sidebar.slider("Number of reviews", 100, 500000, 10000, step=1000)

# Ensemble configuration
st.sidebar.subheader("Ensemble Models")
enable_ensembles = st.sidebar.checkbox("Test Ensemble Models", value=True)
if enable_ensembles:
    ensemble_method = st.sidebar.radio(
        "Ensemble Method",
        ["voting", "averaging"],
        help="Voting: Majority vote | Averaging: Average probabilities"
    )

# Start button
start_button = st.sidebar.button("Start Distributed Attack Simulation",
                                 type="primary",
                                 disabled=not uploaded_file or not loaded_models)

# Main content - System Status
col1, col2, col3, col4 = st.columns(4)

try:
    attacker_count = int(redis_client.get('attacker_count') or 0)
    legit_count = int(redis_client.get('legitimate_count') or 0)
    byzantine_count = int(redis_client.get('byzantine_count') or 0)
    queue_size = redis_client.llen('review_queue')

    col1.metric("Legitimate Users", legit_count)
    col2.metric("Simple Attackers", attacker_count)
    col3.metric("Byzantine Faults", byzantine_count)


    total_users = attacker_count + legit_count + byzantine_count
    if total_users >= 400:
        st.success(f"Users online: {total_users}")
    elif total_users > 0:
        st.warning(f"System initializing... {total_users}/400 users online")
    else:
        st.error("No user containers detected. Run: `docker-compose up -d`")

except Exception as e:
    col1.metric("Legitimate Users", "Error")
    col2.metric("Simple Attackers", "Error")
    col3.metric("Byzantine Faults", "Error")
    col4.metric("Review Queue", "Error")
    st.error(f"Cannot connect to Redis: {e}")

# Main simulation
if uploaded_file and loaded_models:
    # Load data fresh when button is clicked
    if start_button:
        # Load data
        df = load_data_from_file(uploaded_file, sample_size=sample_size)

        # Validate data
        if df is None or len(df) == 0:
            st.error("Failed to load data from uploaded file")
            st.stop()

        st.info(f"Loaded {len(df)} reviews for simulation")

        # Show preview
        with st.expander("Dataset Preview", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reviews", len(df))
            col2.metric("Positive (4-5 stars)", (df['label'] == 1).sum())
            col3.metric("Negative (1-3 stars)", (df['label'] == 0).sum())
            st.dataframe(df.head(10), use_container_width=True)

        st.header("Attack Simulation in Progress")

        # Phase 1: Distribution
        with st.status("Phase 1: Distributing reviews to user containers...", expanded=True) as status:
            total_reviews = distribute_reviews_to_users(df)
            st.write(f"Distributed reviews to Redis queue")
            time.sleep(2)
            status.update(label="Phase 1 Complete", state="complete")

        # Phase 2: Processing
        with st.status("Phase 2: Users processing reviews...", expanded=True) as status:
            st.write("400 concurrent threads processing reviews in parallel...")
            feedback_df = collect_feedback_from_users(len(df), timeout=300)
            st.write(f"Collected {len(feedback_df):,} processed reviews")
            status.update(label="Phase 2 Complete", state="complete")

        # Phase 3: Analysis
        st.header("Attack Analysis")

        poison_count = feedback_df['is_poisoned'].sum()
        poison_rate = (poison_count / len(feedback_df)) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", len(feedback_df))
        col2.metric("Poisoned", poison_count, f"{poison_rate:.1f}%")
        col3.metric("Legitimate", len(feedback_df) - poison_count, f"{100 - poison_rate:.1f}%")

        # Distribution visualization
        fig = px.pie(
            values=[poison_count, len(feedback_df) - poison_count],
            names=['Poisoned', 'Legitimate'],
            title='Review Distribution After Attack',
            color_discrete_sequence=['#ff4444', '#44ff44']
        )
        st.plotly_chart(fig, use_container_width=True)

        # User activity
        with st.expander("User Activity Breakdown", expanded=True):
            user_type_stats = feedback_df.groupby('user_type').agg({
                'user_id': 'nunique',
                'is_poisoned': 'sum'
            }).rename(columns={'user_id': 'unique_users', 'is_poisoned': 'poisoned_reviews'})
            user_type_stats['total_reviews'] = feedback_df.groupby('user_type').size()
            user_type_stats['poison_rate'] = (
                    user_type_stats['poisoned_reviews'] / user_type_stats['total_reviews'] * 100).round(2)

            st.dataframe(user_type_stats, use_container_width=True)

            # Byzantine behavior breakdown if Byzantine users exist
            if 'byzantine' in feedback_df['user_type'].values:
                st.subheader("Byzantine Fault Analysis")

                byzantine_df = feedback_df[feedback_df['user_type'] == 'byzantine']

                if 'byzantine_behavior' in byzantine_df.columns:
                    behavior_stats = byzantine_df['byzantine_behavior'].value_counts()

                    fig_behavior = px.pie(
                        values=behavior_stats.values,
                        names=behavior_stats.index,
                        title='Byzantine Attack Behaviors Distribution'
                    )
                    st.plotly_chart(fig_behavior, use_container_width=True)

                    # Show behavior descriptions
                    st.markdown("""
                    **Byzantine Behavior Types:**
                    - **Inconsistent**: Gives random labels regardless of content
                    - **Conflicting**: Sends multiple contradictory responses for same review
                    - **Delayed Malicious**: Acts legitimate initially, then turns malicious
                    - **Random Corrupt**: Injects random noise into text
                    - **Strategic**: Targets specific rating ranges for maximum impact
                    """)

            # Top active users
            st.subheader("Most Active Users")
            top_users = feedback_df.groupby('user_id').size().sort_values(ascending=False).head(10)
            fig_users = px.bar(
                x=top_users.index,
                y=top_users.values,
                title='Top 10 Most Active Users',
                labels={'x': 'User ID', 'y': 'Reviews Processed'}
            )
            st.plotly_chart(fig_users, use_container_width=True)

        # Phase 4: Model Evaluation
        st.header("Model Evaluation: Clean vs Poisoned Data")

        with st.status("Testing models with poisoned data...", expanded=True) as status:
            results = {}

            # Test individual models
            for model_name, model_info in loaded_models.items():
                st.write(f"Testing {model_name}...")

                # Clean data predictions
                clean_preds, _ = predict_with_model(model_info, df['text'], model_name)
                clean_accuracy = (clean_preds == df['label'].values).mean()

                # Poisoned data predictions
                poisoned_preds, _ = predict_with_model(model_info, feedback_df['text'], model_name)
                poisoned_accuracy = (poisoned_preds == feedback_df['label'].values).mean()

                degradation = (clean_accuracy - poisoned_accuracy) * 100

                results[model_name] = {
                    'clean_accuracy': clean_accuracy,
                    'poisoned_accuracy': poisoned_accuracy,
                    'degradation': degradation
                }

                st.write(f"{model_name}: {clean_accuracy:.3f} -> {poisoned_accuracy:.3f} (-{degradation:.1f}%)")

            # Test ensemble models if enabled
            if enable_ensembles and len(loaded_models) >= 2:
                st.write("---")
                st.write("Testing Ensemble Models...")

                # Define ensemble combinations
                ensembles = {}

                model_names = list(loaded_models.keys())

                # All 2-model combinations
                for combo in combinations(model_names, 2):
                    ensemble_name = f"Ensemble: {combo[0]} + {combo[1]}"
                    ensembles[ensemble_name] = {
                        combo[0]: loaded_models[combo[0]],
                        combo[1]: loaded_models[combo[1]]
                    }

                # All 3-model combinations (if 3 or more models exist)
                if len(loaded_models) >= 3:
                    for combo in combinations(model_names, 3):
                        ensemble_name = f"Ensemble: {combo[0]} + {combo[1]} + {combo[2]}"
                        ensembles[ensemble_name] = {
                            combo[0]: loaded_models[combo[0]],
                            combo[1]: loaded_models[combo[1]],
                            combo[2]: loaded_models[combo[2]]
                        }

                # All 4 models ensemble (if all 4 exist)
                if len(loaded_models) == 4:
                    ensembles['Ensemble: All 4 Models'] = loaded_models.copy()

                # Test each ensemble
                for ensemble_name, ensemble_models in ensembles.items():
                    st.write(f"Testing {ensemble_name}...")

                    # Clean data predictions
                    clean_preds, _ = predict_with_ensemble(
                        ensemble_models,
                        df['text'],
                        ensemble_type=ensemble_method
                    )
                    clean_accuracy = (clean_preds == df['label'].values).mean()

                    # Poisoned data predictions
                    poisoned_preds, _ = predict_with_ensemble(
                        ensemble_models,
                        feedback_df['text'],
                        ensemble_type=ensemble_method
                    )
                    poisoned_accuracy = (poisoned_preds == feedback_df['label'].values).mean()

                    degradation = (clean_accuracy - poisoned_accuracy) * 100

                    results[ensemble_name] = {
                        'clean_accuracy': clean_accuracy,
                        'poisoned_accuracy': poisoned_accuracy,
                        'degradation': degradation
                    }

                    st.write(f"{ensemble_name}: {clean_accuracy:.3f} -> {poisoned_accuracy:.3f} (-{degradation:.1f}%)")

            status.update(label="Model Evaluation Complete", state="complete")

        # Results visualization
        tab1, tab2, tab3 = st.tabs(["Comparison", "Degradation", "Details"])

        with tab1:
            comparison_data = []
            for model_name, metrics in results.items():
                comparison_data.extend([
                    {'Model': model_name, 'Condition': 'Clean', 'Accuracy': metrics['clean_accuracy']},
                    {'Model': model_name, 'Condition': 'Poisoned', 'Accuracy': metrics['poisoned_accuracy']}
                ])

            comparison_df = pd.DataFrame(comparison_data)
            fig = px.bar(
                comparison_df,
                x='Model',
                y='Accuracy',
                color='Condition',
                barmode='group',
                title='Model Performance: Clean vs Poisoned Data',
                color_discrete_map={'Clean': '#44ff44', 'Poisoned': '#ff4444'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            degradation_data = pd.DataFrame([
                {'Model': model, 'Degradation (%)': metrics['degradation']}
                for model, metrics in results.items()
            ])

            fig = px.bar(
                degradation_data,
                x='Model',
                y='Degradation (%)',
                title='Accuracy Degradation Due to Data Poisoning',
                color='Degradation (%)',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            details_df = pd.DataFrame([
                {
                    'Model': model,
                    'Clean Accuracy': f"{metrics['clean_accuracy']:.4f}",
                    'Poisoned Accuracy': f"{metrics['poisoned_accuracy']:.4f}",
                    'Degradation': f"{metrics['degradation']:.2f}%"
                }
                for model, metrics in results.items()
            ])
            st.dataframe(details_df, use_container_width=True)

        # Download results
        st.download_button(
            "Download Results (JSON)",
            json.dumps({
                'timestamp': datetime.now().isoformat(),
                'dataset_size': len(df),
                'poison_rate': poison_rate,
                'model_results': results
            }, indent=2),
            f"attack_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )

        st.success("Attack simulation complete!")

    else:
        # Show preview before simulation starts
        df = load_data_from_file(uploaded_file, sample_size=sample_size)

        with st.expander("Dataset Preview", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reviews", len(df))
            col2.metric("Positive (4-5 stars)", (df['label'] == 1).sum())
            col3.metric("Negative (1-3 stars)", (df['label'] == 0).sum())
            st.dataframe(df.head(10), use_container_width=True)

elif uploaded_file:
    st.info("Please train models first using the provided training scripts")
else:
    st.info("Upload a dataset to begin the simulation")

