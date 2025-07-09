"""
SageMaker Model Deployment Lambda
"""

import json
import boto3
import time
import os


def handler(event, context):
    """Deploy the latest approved model as a SageMaker endpoint"""
    
    print("Starting model deployment")
    
    try:
        # Initialize SageMaker client
        sagemaker = boto3.client('sagemaker')
        
        # Get environment variables
        model_package_group_name = os.environ['MODEL_PACKAGE_GROUP_NAME']
        endpoint_name = os.environ['ENDPOINT_NAME']
        endpoint_config_name = os.environ['ENDPOINT_CONFIG_NAME']
        execution_role_arn = os.environ.get('SAGEMAKER_EXECUTION_ROLE_ARN')
        
        # Step 1: Get latest approved model package
        response = sagemaker.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        
        if not response['ModelPackageSummaryList']:
            return {
                'statusCode': 404,
                'body': json.dumps('No approved model packages found')
            }
        
        model_package_arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']
        print(f"Found model package: {model_package_arn}")
        
        # Step 2: Create SageMaker model
        timestamp = int(time.time())
        model_name = f"iris-model-{timestamp}"
        
        # Get execution role if not provided
        if not execution_role_arn:
            execution_role_arn = get_execution_role()
        
        sagemaker.create_model(
            ModelName=model_name,
            Containers=[{
                'ModelPackageName': model_package_arn
            }],
            ExecutionRoleArn=execution_role_arn
        )
        print(f"Model created: {model_name}")
        
        # Step 3: Clean up existing resources
        cleanup_existing_resources(sagemaker, endpoint_name, endpoint_config_name)
        
        # Step 4: Create endpoint configuration
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'primary',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.t2.medium',
                'InitialVariantWeight': 1.0
            }]
        )
        print(f"Endpoint config created: {endpoint_config_name}")
        
        # Step 5: Create endpoint
        response = sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        endpoint_arn = response['EndpointArn']
        print(f"Endpoint creation started: {endpoint_arn}")
        print("Endpoint will be ready in 10-15 minutes")
        
        return {
            'statusCode': 202,
            'body': json.dumps({
                'message': 'Endpoint creation initiated successfully',
                'status': 'Creating',
                'endpointName': endpoint_name,
                'endpointArn': endpoint_arn,
                'estimatedReadyTime': '10-15 minutes'
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }


def cleanup_existing_resources(sagemaker, endpoint_name, endpoint_config_name):
    """Clean up existing endpoint and config if they exist"""
    
    # Delete endpoint if exists
    try:
        sagemaker.describe_endpoint(EndpointName=endpoint_name)
        print(f"Deleting existing endpoint: {endpoint_name}")
        sagemaker.delete_endpoint(EndpointName=endpoint_name)
        
        # Wait for deletion
        waiter = sagemaker.get_waiter('endpoint_deleted')
        waiter.wait(EndpointName=endpoint_name)
        print("Endpoint deleted")
        
    except sagemaker.exceptions.ClientError:
        print("No existing endpoint to delete")
    
    # Delete endpoint config if exists
    try:
        sagemaker.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"Deleting existing endpoint config: {endpoint_config_name}")
        sagemaker.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        print("Endpoint config deleted")
        
    except sagemaker.exceptions.ClientError:
        print("No existing endpoint config to delete")


def get_execution_role():
    """Get SageMaker execution role ARN"""
    
    iam = boto3.client('iam')
    sts = boto3.client('sts')
    
    # Try to find the pipeline execution role
    role_patterns = [
        'MpsGroupPipelineStack-SageMakerPipelineExecutionRole',
        'SageMakerPipelineExecutionRole'
    ]
    
    for pattern in role_patterns:
        try:
            response = iam.get_role(RoleName=pattern)
            return response['Role']['Arn']
        except:
            continue
    
    # Fallback to default
    account_id = sts.get_caller_identity()['Account']
    return f"arn:aws:iam::{account_id}:role/service-role/AmazonSageMaker-ExecutionRole"
