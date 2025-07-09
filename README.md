# MPS Group SageMaker ML Pipeline 

A fully automated SageMaker ML pipeline using AWS CDK that preprocesses data, trains an XGBoost model, registers it in the SageMaker Model Registry, and deploys it as an inference endpoint.

##  Prerequisites

**direnv** - For predeplopment configuration
   ```bash
   # macOS
   brew install direnv
   ```

##  Configuration

### AWS Configuration (config/test.yaml)
```yaml
bucket_name: mps-group-configured-bucket
versioned: true
removal_policy: DESTROY
dataset_url: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
dataset_key: iris/iris.csv

aws:
  account: "account"
  region: "us-east-2"

sagemaker_domain:
  domain_name: "iris-domain"
  user_profile_name: "default-user"
```

##  Quick Start

1. **Clone and setup**
   ```bash
   git clone https://github.com/paulus01/mps_group
   cd mps_group
   cp .envrc.tmpl .envrc # Paste in .envrc your AWS account ID
   direnv allow  # This will activate the .envrc file
   ```

2. **Deploy infrastructure**
   ```bash
   cdk deploy --all
   ```

## Architecture

### Pipeline Overview
The ML pipeline consists of four main steps:
1. **Data Preprocessing**  - Processes the Iris dataset and splits into train/test sets
2. **Model Training**  - Trains XGBoost classifier using SageMaker training job  
3. **Model Registration**  - Registers trained model in SageMaker Model Registry
4. **Model Deployment**  - Automatically deploys latest approved model as endpoint

### AWS Resources Created
- **S3 Bucket**: Stores datasets and model artifacts
- **VPC**: Secure network infrastructure for SageMaker jobs
- **SageMaker Pipeline**: Orchestrates the ML workflow
- **Model Registry**: Manages model versions and approval workflow
- **Lambda Functions**: Handles automated deployment and dataset upload
- **EventBridge**: Triggers deployment on model registration

##  Deployment Process

### Step 1: S3 Stack
- Creates S3 bucket for storing datasets and model artifacts
- Deploys Lambda function that automatically uploads Iris dataset

### Step 2: VPC Stack  
- Sets up VPC with subnets
- Configures security groups for SageMaker access
- Creates NAT gateway for secure internet access

### Step 3: Pipeline Stack
- Deploys the complete SageMaker pipeline
- Creates Model Registry and approval workflow
- Sets up automated deployment via EventBridge and Lambda

### Automated Workflow
1. **Preprocessing Step**: Processes Iris dataset and splits into train/validation sets
2. **Training Step**: Builds XGBoost model using the processed data
3. **Registration Step**: Registers model in Model Registry with "Approved" status
4. **Deployment Trigger**: EventBridge detects new model registration
5. **Automatic Deployment**: Lambda function creates SageMaker endpoint
   - Monitoring: Automatic cleanup of existing endpoints before deployment

##  Testing and Validation

### Endpoint Testing
```bash
python test/test_endpoint.py
```