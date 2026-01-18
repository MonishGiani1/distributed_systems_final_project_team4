# Distributed Data Poisoning Attack Simulation

A comprehensive demonstration platform for studying data poisoning attacks in distributed machine learning systems. This project simulates a federated learning environment where 400 concurrent users (legitimate and malicious) provide feedback that influences ML model training.

##  Overview

This simulation demonstrates:
- **Data Poisoning Attacks**: Label flipping and Byzantine fault attacks
- **Multi-Model Testing**: Logistic Regression, SVM, SBERT, DistilBERT
- **Ensemble Methods**: Voting and averaging strategies
- **Spectral Defense**: Advanced poisoning detection using spectral signatures
- **Real-time Visualization**: Interactive dashboards showing attack impacts

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Azure Cloud Deployment](#azure-cloud-deployment)
- [Usage Guide](#usage-guide)
- [Model Training](#model-training)
- [Understanding Results](#understanding-results)
- [Troubleshooting](#troubleshooting)

## Features

### Attack Simulation
- **400 Concurrent Users**: 200 legitimate, 150 attackers, 50 Byzantine
- **Real-time Processing**: Redis-based message queuing
- **Multiple Attack Types**: 
  - Label flipping (simple attackers)
  - Byzantine faults (inconsistent, conflicting, delayed attacks)

### Machine Learning Models
- **Traditional ML**: Logistic Regression, SVM
- **Deep Learning**: SBERT, DistilBERT
- **Ensemble Methods**: Voting and averaging strategies

### Defense Mechanisms
- **Spectral Signature Filtering**: Detects poisoned samples using PCA-based analysis
- **Configurable Thresholds**: Adjust sensitivity and contamination rates
- **Real-time Filtering**: Apply defense during simulation

### Visualization
- Attack distribution analysis
- Model performance comparisons
- Attack Success Rate (ASR) metrics
- User activity breakdown

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Web App                    │
│              (Model Training & Evaluation)              │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Redis Queue   │
         │  (Messaging)   │
         └────┬───────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
    ▼         ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐
│Legit   │ │Attack  │ │Byzant  │
│Pool    │ │Pool    │ │Pool    │
│(200)   │ │(150)   │ │(50)    │
└────────┘ └────────┘ └────────┘
```

## Prerequisites

### For Local Deployment
- **Docker Desktop** (latest version)
- **Docker Compose** (v2.0+)
- **Python 3.8+** (for model training)
- **8GB RAM minimum** (16GB recommended)
- **NVIDIA GPU** (optional, for deep learning training)

### For Azure Cloud Deployment
- **Azure Account** (Azure for Students or standard subscription)
- **Azure CLI** installed
- **Docker** installed
- **Docker Hub** account (free)

## Local Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/distributed_systems_FP.git
cd distributed_systems_FP
```

### Step 2: Prepare Dataset

Create a CSV file with the following columns:
- `text`: Review text
- `rating`: Star rating (1-5)

Example structure:
```csv
text,rating
"Great product! Highly recommend.",5
"Terrible quality. Waste of money.",1
```

Place your dataset in the `data/` directory:
```bash
mkdir -p data
```

### Step 3: Start Docker Containers

```bash
# Start all services
docker-compose up -d

# Verify containers are running
docker-compose ps

# You should see 5 containers:
# - streamlit-app
# - redis
# - legitimate-pool
# - attacker-pool
# - byzantine-pool
```

### Step 4: Access Application

Open browser and navigate to:
```
http://localhost:8501
```

Wait 30-60 seconds for all 400 user threads to initialize.

### Step 5: Run Simulation

1. **Upload Dataset**: Click "Upload Dataset (CSV)" in sidebar
2. **Configure Settings**: 
   - Adjust sample size (default: 10,000)
   - Enable/disable spectral defense
   - Configure ensemble methods
3. **Start Simulation**: Click "Start Distributed Attack Simulation"
4. **Monitor Progress**: Watch real-time processing of reviews
5. **View Results**: Analyze attack impacts and model performance

## Azure Cloud Deployment

### Prerequisites
- Azure subscription (Azure for Students recommended)
- Docker Hub account
- Azure CLI installed

### Automated Deployment Script

```bash
chmod +x azure-setup.sh

./azure-setup.sh
```

The script will:
1. Login to Azure
2. Create resource group
3. Build Docker images
4. Push to Docker Hub
5. Deploy to Azure Container Instances
6. Provide access URL

### Manual Deployment Steps

If you prefer manual deployment:

#### 1. Login to Azure

```bash
az login
```

#### 2. Create Resource Group

```bash
RESOURCE_GROUP="poison-simulation-rg"
LOCATION="eastus"  

az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION
```

#### 3. Build and Push Docker Images

```bash
docker login

DOCKERHUB_USERNAME="yourusername"

docker build -f Dockerfile.streamlit -t ${DOCKERHUB_USERNAME}/poison-streamlit:latest .
docker build -f Dockerfile.legitimate -t ${DOCKERHUB_USERNAME}/poison-legitimate:latest .
docker build -f Dockerfile.attacker -t ${DOCKERHUB_USERNAME}/poison-attacker:latest .
docker build -f Dockerfile.byzantine -t ${DOCKERHUB_USERNAME}/poison-byzantine:latest .

docker push ${DOCKERHUB_USERNAME}/poison-streamlit:latest
docker push ${DOCKERHUB_USERNAME}/poison-legitimate:latest
docker push ${DOCKERHUB_USERNAME}/poison-attacker:latest
docker push ${DOCKERHUB_USERNAME}/poison-byzantine:latest
```

#### 4. Update Deployment File

Edit `azure-deploy.yaml` and replace image names:

```yaml
image: yourusername/poison-streamlit:latest
image: yourusername/poison-legitimate:latest
image: yourusername/poison-attacker:latest
image: yourusername/poison-byzantine:latest
```

#### 5. Deploy Container Group

```bash
az container create \
  --resource-group $RESOURCE_GROUP \
  --file azure-deploy.yaml
```

#### 6. Get Public URL

```bash
az container show \
  --resource-group $RESOURCE_GROUP \
  --name poison-simulation \
  --query ipAddress.fqdn \
  --output tsv
```

Access your app at: `http://<fqdn>:8501`

### Azure Management Commands

**View logs:**
```bash
az container logs \
  --resource-group poison-simulation-rg \
  --name poison-simulation \
  --container-name streamlit-app
```

**Check status:**
```bash
az container show \
  --resource-group poison-simulation-rg \
  --name poison-simulation \
  --output table
```

**Stop containers (keeps resources):**
```bash
az container stop \
  --resource-group poison-simulation-rg \
  --name poison-simulation
```

**Start containers:**
```bash
az container start \
  --resource-group poison-simulation-rg \
  --name poison-simulation
```

**Delete everything:**
```bash
az group delete \
  --name poison-simulation-rg \
  --yes --no-wait
```

### Cost Optimization

Azure Container Instances pricing (approximate):
- **Total Resources**: 3 vCPUs, 4GB RAM
- **Estimated Cost**: ~$0.15-0.25/hour
- **Daily Cost**: ~$3.60-6.00/day

**Recommendations:**
- Stop containers when not in use
- Delete resource group after demos
- Use Azure for Students credits ($100 free)

## Usage Guide

### Basic Workflow

1. **Prepare Dataset**
   - CSV format with `text` and `rating` columns
   - Minimum 1,000 reviews recommended
   - Maximum 50,000 reviews (for performance)

2. **Upload Dataset**
   - Click sidebar upload button
   - Select CSV file
   - Wait for validation

3. **Configure Simulation**
   - **Sample Size**: Number of reviews to process (10-50k)
   - **Spectral Defense**: Enable/disable poisoning detection
   - **Ensemble Models**: Test model combinations

4. **Start Simulation**
   - Click "Start Distributed Attack Simulation"
   - Watch real-time progress
   - Wait for completion (2-5 minutes)

5. **Analyze Results**
   - View accuracy degradation
   - Check Attack Success Rates
   - Compare model performance
   - Download results (JSON)

### Spectral Defense Configuration

**Expected Attack Rate** (contamination):
- Set to actual attacker proportion (default: 0.375 = 37.5%)
- Higher values = more aggressive filtering
- Lower values = miss some attacks

**Detection Threshold** (percentile):
- 90-95: Balanced detection
- 95-99: Conservative (fewer false positives)
- Higher = stricter filtering

### Understanding Metrics

**Accuracy**:
- Clean Accuracy: Performance on unpoisoned data
- Poisoned Accuracy: Performance after attack
- Degradation: Accuracy loss (%)

**Attack Success Rate (ASR)**:
- Overall: Success across all attacks
- Label Flip: Success on simple flipping attacks
- Byzantine: Success on complex fault attacks
- **Lower ASR = Better defense**

## Model Training

### Traditional Models (CPU)

```bash
python train_traditional_models.py
```

**Configuration:**
- Dataset: 100k samples
- Models: Logistic Regression, SVM
- Training Time: 10-20 minutes (CPU)
- Output: `models/logistic_regression_*.pkl`, `models/svm_*.pkl`

### Deep Learning Models (GPU)

```bash
python train_modern_models.py
```

**Configuration:**
- Dataset: 100k samples
- Models: DistilBERT, SBERT
- Training Time: 2-3 hours (RTX 3050)
- Output: `models/distilbert/`, `models/sbert/`

**GPU Requirements:**
- NVIDIA GPU with 4GB+ VRAM
- CUDA 11.0+
- Mixed precision (fp16) enabled



## Understanding Results

### Typical Results (Without Defense)

| Model | Clean Acc | Poisoned Acc | Degradation | ASR |
|-------|-----------|--------------|-------------|-----|
| Logistic Regression | 0.910 | 0.710 | 20.0% | 15.5% |
| SVM | 0.920 | 0.720 | 20.0% | 12.5% |
| SBERT | 0.830 | 0.660 | 17.0% | 19.5% |
| DistilBERT | 0.870 | 0.720 | 15.0% | 23.0% |

### With Spectral Defense

| Model | Poisoned Acc | Degradation | ASR | Filtered |
|-------|--------------|-------------|-----|----------|
| Logistic Regression | 0.850 | 6.0% | 8.2% | 26% |
| SVM | 0.880 | 4.0% | 6.5% | 26% |
| SBERT | 0.790 | 4.0% | 10.1% | 26% |
| DistilBERT | 0.840 | 3.0% | 12.8% | 26% |

**Key Insights:**
- Spectral defense reduces degradation by ~70-80%
- ASR drops significantly (40-50% reduction)
- ~26% of samples filtered (matches attack rate)

## Troubleshooting

### "No user containers detected"

**Problem**: User pools not starting

**Solution**:
```bash
docker-compose down
docker-compose up -d

docker-compose logs legitimate-pool
docker-compose logs attacker-pool
docker-compose logs byzantine-pool
```

### "Cannot connect to Redis"

**Problem**: Redis not accessible

**Solution**:
```bash
docker-compose ps redis

docker-compose restart redis

docker-compose exec redis redis-cli ping
```

### Slow Performance

**Problem**: Simulation takes too long

**Solutions**:
1. Reduce sample size (sidebar slider)
2. Increase Docker memory allocation (Docker Desktop settings)
3. Use smaller dataset
4. Disable ensemble models

### Azure Deployment Fails

**Problem**: Container group creation errors

**Solutions**:

1. **Check resource limits:**
```bash
az vm list-usage --location eastus --output table
```

2. **Reduce resources in `azure-deploy.yaml`:**
```yaml
cpu: 0.5  
memoryInGb: 1.0  
```

3. **Check Docker Hub images:**
```bash
docker search yourusername/poison-streamlit
```

### Models Not Loading

**Problem**: "No pre-trained models found"

**Solutions**:

1. **Train models locally:**
```bash
python train_traditional_models.py
python train_modern_models.py
```

2. **Check models directory:**
```bash
ls -la models/
```

3. **Download from Azure:**
   - Models auto-download on first run
   - Check Azure Storage credentials in `azure-deploy.yaml`

## Project Structure

```
distributed_systems_FP/
├── streamlit_app.py           # Main web application
├── spectral_defense.py        # Defense mechanism implementation
├── legitimate_pool.py         # Legitimate user simulator
├── attacker_pool.py          # Attacker user simulator
├── byzantine_pool.py         # Byzantine fault simulator
├── train_traditional_models.py  # LR + SVM training
├── train_modern_models.py    # SBERT + DistilBERT training
├── docker-compose.yml        # Local deployment config
├── azure-deploy.yaml         # Azure deployment config
├── azure-setup.sh            # Automated Azure deployment
├── requirements.txt          # Python dependencies
├── Dockerfile.streamlit      # Streamlit container
├── Dockerfile.legitimate     # Legitimate pool container
├── Dockerfile.attacker       # Attacker pool container
├── Dockerfile.byzantine      # Byzantine pool container
├── models/                   # Trained model files
│   ├── logistic_regression_*.pkl
│   ├── svm_*.pkl
│   ├── sbert/
│   └── distilbert/
└── data/                     # Dataset directory
    └── your_dataset.csv
```


## Acknowledgments

- **Datasets**: Amazon Product Reviews
- **Models**: HuggingFace Transformers, Sentence-Transformers
- **Infrastructure**: Docker, Redis, Azure Container Instances
- **Visualization**: Streamlit, Plotly



