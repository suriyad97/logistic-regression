# Deploying to Existing Azure ML Infrastructure

Since your Azure ML workspace and blob storage are already set up, follow these steps to deploy the pipeline.

## Prerequisites

- ✅ Azure ML Workspace created
- ✅ Blob storage configured
- ✅ Compute cluster provisioned
- ✅ Resource group with proper permissions
- Azure CLI installed and authenticated

## Quick Deployment (3 steps)

### Step 1: Authenticate with Azure

```bash
az login
az account set --subscription <YOUR_SUBSCRIPTION_ID>
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run Deployment Script

```bash
python scripts/deploy.py \
    --subscription-id <YOUR_SUBSCRIPTION_ID> \
    --resource-group <YOUR_RESOURCE_GROUP> \
    --workspace-name <YOUR_WORKSPACE_NAME>
```

This will automatically:
1. Upload titanic.csv to blob storage
2. Submit the training pipeline to Azure ML
3. Wait for pipeline completion
4. Show next steps for API deployment

## Manual Deployment Steps

If you prefer to run each step individually:

### Step 1: Setup and Verify Workspace Connection

```bash
python scripts/setup_workspace.py
```

This will:
- Connect to your Azure ML workspace
- Verify compute cluster availability
- Create/verify the environment
- Upload data to blob storage

### Step 2: Upload Data to Blob Storage

```bash
python scripts/upload_data.py \
    --subscription-id <SUBSCRIPTION_ID> \
    --resource-group <RESOURCE_GROUP> \
    --workspace-name <WORKSPACE_NAME> \
    --file titanic.csv
```

### Step 3: Train Locally (Optional - for testing)

```bash
python src/train.py \
    --data titanic.csv \
    --output ./model \
    --test-size 0.2
```

### Step 4: Evaluate Model

```bash
python src/evaluate.py \
    --model ./model/model.pkl \
    --preprocessor ./model/preprocessor.pkl \
    --data titanic.csv \
    --output ./evaluation
```

### Step 5: Submit Pipeline to Azure ML

```bash
python scripts/submit_pipeline.py \
    --subscription-id <SUBSCRIPTION_ID> \
    --resource-group <RESOURCE_GROUP> \
    --workspace-name <WORKSPACE_NAME> \
    --wait
```

### Step 6: Deploy API

**Option A: Run locally**
```bash
python scripts/app.py
```

**Option B: Docker**
```bash
# Build image
docker build -t titanic:latest .

# Run container
docker run -p 5000:5000 -v $(pwd)/model:/model titanic:latest
```

**Option C: Docker Compose**
```bash
docker-compose up
```

## Monitoring Training

### In Azure ML Studio

1. Go to your workspace: `https://ml.azure.com`
2. Click on "Experiments"
3. Find "titanic-logistic-regression-exp"
4. View job status, metrics, and logs

### Using Azure CLI

```bash
# List recent runs
az ml job list --workspace-name <WORKSPACE_NAME> --resource-group <RESOURCE_GROUP>

# Get specific run details
az ml job show --name <JOB_ID> --workspace-name <WORKSPACE_NAME>

# Stream job logs
az ml job stream --name <JOB_ID> --workspace-name <WORKSPACE_NAME>
```

## Environment Variables

Create a `.env` file in the project root:

```bash
cp config/.env.example .env
```

Edit `.env` with your values:

```
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_RESOURCE_GROUP=your_resource_group
AZURE_ML_WORKSPACE=your_workspace_name
```

Then you can use:

```bash
python scripts/deploy.py  # Will read from .env
```

## Troubleshooting

### Issue: Authentication Error

```bash
# Clear cached credentials
az logout
az cache purge

# Re-authenticate
az login
```

### Issue: Workspace Not Found

```bash
# Verify workspace exists
az ml workspace show \
    --name <WORKSPACE_NAME> \
    --resource-group <RESOURCE_GROUP>
```

### Issue: Compute Cluster Not Ready

```bash
# Check compute status
az ml compute show \
    --name cpu-cluster \
    --workspace-name <WORKSPACE_NAME>
```

### Issue: Data Upload Fails

```bash
# Verify storage account exists
az storage account show \
    --name <STORAGE_ACCOUNT_NAME> \
    --resource-group <RESOURCE_GROUP>

# Check storage keys
az storage account keys list \
    --account-name <STORAGE_ACCOUNT_NAME> \
    --resource-group <RESOURCE_GROUP>
```

## API Testing

Once deployed, test the API:

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

# Batch predictions
curl -X POST http://localhost:5000/batch-predict \
  -H "Content-Type: application/json" \
  -d '[
    {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"},
    {"Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0, "Fare": 71.28, "Embarked": "C"}
  ]'
```

## Expected Pipeline Output

After pipeline completion, you'll have:

1. **Trained Model**
   - `model/model.pkl` - Serialized model
   - `model/preprocessor.pkl` - Feature preprocessor
   - `model/metrics.json` - Training metrics

2. **Evaluation Results**
   - `evaluation/evaluation_metrics.json` - Test metrics
   - `evaluation/roc_curve.png` - ROC visualization
   - `evaluation/confusion_matrix.png` - Confusion matrix

3. **Registered Model** (in Azure ML registry)
   - Version: 1.0.0
   - Can be used for further deployments

## Next Steps

1. **Monitor Performance**
   - Set up alerts in Application Insights
   - Track model metrics over time

2. **Deploy to Production**
   - Deploy API to Azure Container Instances
   - Deploy to Azure App Service
   - Use Azure Functions for serverless inference

3. **Set Up Retraining**
   - Schedule periodic pipeline runs
   - Set up data drift detection
   - Implement automated retraining

4. **Advanced Monitoring**
   - Add data validation checks
   - Implement model validation gates
   - Set up automated model registry

## Example: Full Automated Deployment

```bash
#!/bin/bash

SUBSCRIPTION_ID="your_subscription_id"
RESOURCE_GROUP="your_resource_group"
WORKSPACE_NAME="your_workspace_name"

# Install dependencies
pip install -r requirements.txt

# Authenticate
az login
az account set --subscription $SUBSCRIPTION_ID

# Run complete deployment
python scripts/deploy.py \
    --subscription-id $SUBSCRIPTION_ID \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $WORKSPACE_NAME

# Start API
python scripts/app.py
```

Save this as `deploy.sh` and run: `bash deploy.sh`

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review Azure ML documentation
3. Check script logs for detailed error messages
4. See main README.md for comprehensive guide

---

**Status**: Ready for Existing Infrastructure
**Estimated Time**: 5-10 minutes for complete deployment
**Success Indicators**: ✓ Data uploaded, ✓ Pipeline submitted, ✓ API running
