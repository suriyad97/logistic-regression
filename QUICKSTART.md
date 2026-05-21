# Quick Start Guide - Titanic Logistic Regression

Get up and running in 5 minutes!

## 1. Clone and Setup

```bash
cd mlops-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Train Model (Local)

```bash
python src/train.py --data titanic.csv --output ./model
```

Expected output:
- `model/model.pkl` - Trained model
- `model/preprocessor.pkl` - Preprocessor
- `model/metrics.json` - Training metrics

## 3. Evaluate Model

```bash
python src/evaluate.py \
    --model ./model/model.pkl \
    --preprocessor ./model/preprocessor.pkl \
    --data titanic.csv \
    --output ./evaluation
```

Output directory contains:
- `evaluation_metrics.json` - Detailed metrics
- `roc_curve.png` - ROC curve visualization
- `confusion_matrix.png` - Confusion matrix

## 4. Run API Server

```bash
# Option A: Direct Python
python scripts/app.py

# Option B: Using Docker
docker build -t titanic:latest .
docker run -p 5000:5000 -v $(pwd)/model:/model titanic:latest

# Option C: Using Docker Compose
docker-compose up
```

## 5. Test API

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
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

## 6. Deploy to Azure (Optional)

### Prerequisites
```bash
# Install Azure CLI
# https://learn.microsoft.com/cli/azure/install-azure-cli

# Login to Azure
az login

# Set default subscription
az account set --subscription <SUBSCRIPTION_ID>
```

### Deploy Infrastructure
```bash
# Create resource group
az group create \
    --name rg-titanic-ml \
    --location eastus

# Deploy resources
az deployment group create \
    --name ml-deployment \
    --resource-group rg-titanic-ml \
    --template-file infra/main.bicep
```

### Submit Pipeline
```bash
python scripts/submit_pipeline.py \
    --subscription-id <SUBSCRIPTION_ID> \
    --resource-group rg-titanic-ml \
    --workspace-name titanic-ml-ws \
    --wait
```

## Key Files

| File | Purpose |
|------|---------|
| `src/data_processing.py` | Data loading and preprocessing |
| `src/train.py` | Model training |
| `src/evaluate.py` | Model evaluation |
| `src/inference.py` | Inference engine |
| `scripts/app.py` | Flask API server |
| `Dockerfile` | Docker image definition |
| `requirements.txt` | Python dependencies |

## Features

✅ Comprehensive data preprocessing with feature engineering
✅ Logistic regression with class balancing
✅ Detailed evaluation metrics and visualizations
✅ REST API for predictions
✅ Docker containerization
✅ Azure ML integration
✅ MLflow experiment tracking
✅ Production-ready error handling

## Troubleshooting

**Model training fails with import error:**
```bash
pip install -r requirements.txt --force-reinstall
```

**Port 5000 already in use:**
```bash
# On Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# On Linux/Mac
lsof -i :5000
kill -9 <PID>
```

**Docker build fails:**
```bash
docker build -t titanic:latest . --progress=plain --no-cache
```

## Next Steps

- Review [README.md](README.md) for comprehensive documentation
- Check [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- Explore Azure ML monitoring in your workspace
- Set up CI/CD pipeline using GitHub Actions

## Support

For detailed information, refer to:
- [Azure ML Documentation](https://learn.microsoft.com/azure/machine-learning/)
- [Scikit-learn Docs](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)

Happy ML engineering! 🚀
