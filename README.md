# Titanic Logistic Regression - Production Ready ML Pipeline

Production-ready machine learning solution for predicting Titanic passenger survival using logistic regression, deployed on Azure Machine Learning.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [API Usage](#api-usage)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a complete MLOps pipeline for:
- **Data Processing**: Feature engineering, handling missing values, encoding categorical features
- **Model Training**: Logistic regression with class balancing
- **Evaluation**: Comprehensive metrics and visualizations
- **Deployment**: Docker containerization and Azure ML deployment
- **Serving**: REST API for batch and single predictions

### Model Performance

- **Accuracy**: ~79-80%
- **Precision**: ~79%
- **Recall**: ~67%
- **F1-Score**: ~72%
- **ROC-AUC**: ~85%

## Project Structure

```
mlops-pipeline/
├── src/
│   ├── data_processing.py      # Data loading, cleaning, feature engineering
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Model evaluation script
│   └── inference.py            # Inference engine for predictions
├── config/
│   ├── config.py               # Configuration classes
│   ├── environment.yml         # Conda environment specification
│   └── .env.example            # Environment variables template
├── scripts/
│   ├── submit_pipeline.py      # Azure ML pipeline submission
│   ├── register_model.py       # Model registration to registry
│   └── app.py                  # Flask API for serving
├── pipelines/
│   └── pipeline.yml            # Azure ML pipeline definition
├── infra/
│   ├── main.bicep              # Azure ML infrastructure (Bicep)
│   └── main-params.bicep       # Bicep parameters
├── titanic.csv                 # Training data
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Local Docker setup
└── README.md                   # This file
```

## Prerequisites

### Local Development
- Python 3.10+
- pip or conda
- Docker (for containerization)

### Azure Deployment
- Azure subscription
- Azure CLI (az) and Azure Developer CLI (azd)
- Appropriate RBAC permissions for:
  - Azure Machine Learning Workspace
  - Storage accounts
  - Container Registry
  - Key Vault

### Authentication
The project uses **Managed Identity** for Azure authentication:
- When running in Azure (Azure ML Compute): Automatic via system-assigned identity
- Local development: Uses Azure CLI credentials via `DefaultAzureCredential`

## Local Setup

### 1. Clone Repository and Install Dependencies

```bash
cd mlops-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Environment Variables

```bash
cp config/.env.example .env
# Edit .env with your Azure credentials
```

### 3. Verify Data

```bash
python -c "import pandas as pd; df = pd.read_csv('titanic.csv'); print(f'Data shape: {df.shape}')"
```

## Training

### Local Training

```bash
# Train model
python src/train.py \
    --data titanic.csv \
    --output ./model \
    --test-size 0.2 \
    --max-iter 1000

# Output files:
# - model/model.pkl           - Trained model
# - model/preprocessor.pkl    - Feature preprocessor
# - model/metrics.json        - Training metrics
```

### With MLflow Tracking

The training script automatically logs to MLflow if available:

```bash
# View MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Access at http://localhost:5000
```

### Azure ML Training

```bash
# Submit to Azure ML
python scripts/submit_pipeline.py \
    --subscription-id <YOUR_SUBSCRIPTION_ID> \
    --resource-group <YOUR_RESOURCE_GROUP> \
    --workspace-name <YOUR_WORKSPACE_NAME> \
    --wait
```

## Evaluation

### Local Evaluation

```bash
python src/evaluate.py \
    --model model/model.pkl \
    --preprocessor model/preprocessor.pkl \
    --data titanic.csv \
    --output ./evaluation

# Output files:
# - evaluation/evaluation_metrics.json
# - evaluation/roc_curve.png
# - evaluation/confusion_matrix.png
```

### Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Deployment

### 1. Deploy Infrastructure

```bash
# Deploy Azure ML infrastructure
az deployment group create \
    --name ml-deployment \
    --resource-group <YOUR_RESOURCE_GROUP> \
    --template-file infra/main.bicep \
    --parameters \
        location=eastus \
        workspaceName=titanic-ml-ws \
        storageAccountName=titanicml \
        keyVaultName=titanic-kv \
        appInsightsName=titanic-insights
```

### 2. Build Docker Image

```bash
# Build locally
docker build -t titanic-logistic-regression:latest .

# Test locally
docker run -p 5000:5000 \
    -v $(pwd)/model:/model \
    titanic-logistic-regression:latest
```

### 3. Push to Container Registry

```bash
# Login to ACR
az acr login --name <YOUR_REGISTRY_NAME>

# Tag and push
docker tag titanic-logistic-regression:latest \
    <YOUR_REGISTRY_NAME>.azurecr.io/titanic-logistic-regression:latest

docker push <YOUR_REGISTRY_NAME>.azurecr.io/titanic-logistic-regression:latest
```

### 4. Deploy to Azure Container Instances or App Service

```bash
# Deploy to Container Instances
az container create \
    --resource-group <YOUR_RESOURCE_GROUP> \
    --name titanic-ml-inference \
    --image <YOUR_REGISTRY_NAME>.azurecr.io/titanic-logistic-regression:latest \
    --ports 5000 \
    --registry-login-server <YOUR_REGISTRY_NAME>.azurecr.io \
    --registry-username <USERNAME> \
    --registry-password <PASSWORD>
```

### 5. Register Model

```bash
python scripts/register_model.py \
    --subscription-id <SUBSCRIPTION_ID> \
    --resource-group <RESOURCE_GROUP> \
    --workspace-name <WORKSPACE_NAME> \
    --model-path model/model.pkl \
    --preprocessor-path model/preprocessor.pkl \
    --model-name titanic-logistic-regression \
    --model-version 1.0.0
```

## API Usage

### Health Check

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Single Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "Did not survive",
  "survival_probability": 0.24,
  "non_survival_probability": 0.76
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:5000/batch-predict \
  -H "Content-Type: application/json" \
  -d '[
    {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"},
    {"Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0, "Fare": 71.28, "Embarked": "C"}
  ]'
```

### Model Information

```bash
curl http://localhost:5000/model-info
```

## Monitoring

### Azure ML Monitoring

1. **Azure ML Studio**
   - Navigate to your workspace
   - View run history and metrics
   - Monitor compute resources

2. **Application Insights**
   - Track API requests and performance
   - View custom metrics and logs

3. **MLflow Tracking**
   - View experiments and runs
   - Compare metrics across runs

### Metrics to Monitor

- **Model Performance**: Accuracy, precision, recall over time
- **Data Drift**: Monitor incoming data distribution changes
- **API Performance**: Response time, throughput, error rates
- **Resource Usage**: CPU, memory, GPU utilization

## Security Best Practices

1. **Authentication**
   - Use Managed Identity in Azure
   - Never hardcode credentials
   - Store secrets in Key Vault

2. **Data Protection**
   - Enable encryption in transit (HTTPS)
   - Enable encryption at rest
   - Implement network isolation

3. **Access Control**
   - Use RBAC for resource access
   - Implement least privilege principle
   - Regularly audit access logs

4. **Model Security**
   - Version models in registry
   - Track model lineage
   - Implement model governance

## Troubleshooting

### Issue: Model Loading Error

```bash
# Check file paths
ls -la model/

# Verify pickle compatibility
python -c "import joblib; joblib.load('model/model.pkl')"
```

### Issue: Azure ML Authentication Error

```bash
# Verify Azure CLI login
az account show

# Verify workspace connectivity
az ml workspace show --name <WORKSPACE_NAME>
```

### Issue: Docker Build Failure

```bash
# Build with verbose output
docker build -t titanic:latest . --progress=plain

# Check dependencies
pip list
```

### Issue: API Port Already in Use

```bash
# On Linux/Mac
lsof -i :5000
kill -9 <PID>

# On Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

## Performance Optimization

### Model Optimization
- Hyperparameter tuning (max_iter, solver, class_weight)
- Feature selection and dimensionality reduction
- Batch normalization and scaling

### API Optimization
- Connection pooling for database operations
- Caching frequently accessed predictions
- Load balancing across instances
- Rate limiting

### Infrastructure Optimization
- Auto-scaling for compute resources
- Right-sizing VM instances
- Using GPU for inference when beneficial
- CDN for API distribution

## Continuous Integration/Deployment

GitHub Actions workflow example (`.github/workflows/ml-pipeline.yml`):

```yaml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Azure ML
        run: python scripts/submit_pipeline.py
```

## Contributing

1. Create a feature branch
2. Implement changes with tests
3. Submit pull request with documentation
4. Ensure all tests pass

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- Check troubleshooting section
- Review Azure ML documentation
- Contact ML Ops team

## References

- [Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)
