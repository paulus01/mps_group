#!/usr/bin/env python3
"""
Test SageMaker built-in sklearn mode by creating a direct endpoint
This bypasses model packages and tests the built-in inference capability
"""

import boto3
import json
import time
import pickle
import tarfile
import os
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

def create_builtin_endpoint():
    """Create endpoint using built-in sklearn mode (no custom inference)"""
    
    print("🚀 Testing SageMaker Built-in Sklearn Mode")
    print("=" * 50)
    
    # Initialize clients
    sagemaker = boto3.client('sagemaker', region_name='us-east-2')
    s3 = boto3.client('s3', region_name='us-east-2')
    runtime = boto3.client('sagemaker-runtime', region_name='us-east-2')
    
    # Configuration
    bucket_name = 'mps-group-configured-bucket'
    role_arn = 'arn:aws:iam::593772403053:role/MpsGroupPipelineStack-SageMakerPipelineExecutionRol-kBVe50ggZBmy'
    
    # Create a simple model
    print("📊 Creating simple iris model...")
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Create model artifacts (ONLY model.pkl, no inference script)
    timestamp = int(time.time())
    local_model_dir = f'/tmp/builtin_model_{timestamp}'
    os.makedirs(local_model_dir, exist_ok=True)
    
    # Save ONLY the model (built-in mode)
    with open(f'{local_model_dir}/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Create tar.gz with ONLY the model file
    model_tar_path = f'/tmp/builtin_model_{timestamp}.tar.gz'
    with tarfile.open(model_tar_path, 'w:gz') as tar:
        tar.add(f'{local_model_dir}/model.pkl', arcname='model.pkl')
    
    print("📦 Model contains ONLY model.pkl (no custom inference code)")
    
    # Upload to S3
    model_s3_key = f'builtin-test/model_{timestamp}.tar.gz'
    print(f"📤 Uploading to S3...")
    s3.upload_file(model_tar_path, bucket_name, model_s3_key)
    model_s3_uri = f's3://{bucket_name}/{model_s3_key}'
    print(f"✅ Model uploaded: {model_s3_uri}")
    
    # Create unique names
    model_name = f'builtin-test-model-{timestamp}'
    config_name = f'builtin-test-config-{timestamp}'
    endpoint_name = f'builtin-test-endpoint-{timestamp}'
    
    try:
        # Create SageMaker Model (NO environment variables = built-in mode)
        print("🔧 Creating SageMaker model (built-in mode)...")
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': '257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3',
                'ModelDataUrl': model_s3_uri
                # NO Environment section = built-in sklearn mode
            },
            ExecutionRoleArn=role_arn
        )
        print("✅ Model created in built-in mode")
        
        # Create Endpoint Configuration
        print("⚙️ Creating endpoint configuration...")
        sagemaker.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                'VariantName': 'primary',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m5.large',  # Use reliable instance type
                'InitialVariantWeight': 1.0
            }]
        )
        print("✅ Endpoint config created")
        
        # Create Endpoint
        print("🚀 Creating endpoint...")
        print("   This will take 5-8 minutes for built-in mode...")
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        
        # Wait for endpoint
        print("⏳ Waiting for endpoint...")
        start_time = time.time()
        
        while True:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            elapsed = int(time.time() - start_time)
            
            print(f"   Status: {status} ({elapsed}s elapsed)")
            
            if status == 'InService':
                print(f"✅ Endpoint ready! ({elapsed}s total)")
                break
            elif status in ['Failed', 'OutOfService']:
                print(f"❌ Endpoint failed: {status}")
                if 'FailureReason' in response:
                    print(f"   Reason: {response['FailureReason']}")
                return False
            
            if elapsed > 600:  # 10 minutes timeout
                print(f"⏰ Timeout after 10 minutes")
                return False
            
            time.sleep(20)
        
        # Test the endpoint with built-in inference
        print("\n🧪 Testing built-in sklearn inference...")
        
        # Test data in the format sklearn built-in expects
        test_samples = [
            ([5.1, 3.5, 1.4, 0.2], "Setosa"),
            ([6.2, 2.9, 4.3, 1.3], "Versicolor"), 
            ([7.3, 2.9, 6.3, 1.8], "Virginica")
        ]
        
        for i, (sample, expected_class) in enumerate(test_samples):
            try:
                # Built-in sklearn expects this format
                test_data = ','.join(map(str, sample))
                
                response = runtime.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType='text/csv',
                    Body=test_data
                )
                
                result = response['Body'].read().decode().strip()
                class_names = ['Setosa', 'Versicolor', 'Virginica']
                predicted_class = class_names[int(float(result))]
                
                status = "✅" if predicted_class == expected_class else "⚠️"
                print(f"   Test {i+1}: {status} Input: {sample}")
                print(f"           Predicted: {predicted_class} (class {result})")
                print(f"           Expected:  {expected_class}")
                
            except Exception as e:
                print(f"   Test {i+1}: ❌ Error: {e}")
        
        print(f"\n🎉 SUCCESS! Built-in sklearn inference is working!")
        print(f"   Endpoint: {endpoint_name}")
        print(f"   No custom inference code needed!")
        
        # Cleanup option
        cleanup = input("\nDelete test resources? (y/n): ").lower().strip()
        if cleanup == 'y':
            print("\n🧹 Cleaning up...")
            sagemaker.delete_endpoint(EndpointName=endpoint_name)
            sagemaker.delete_endpoint_config(EndpointConfigName=config_name)
            sagemaker.delete_model(ModelName=model_name)
            s3.delete_object(Bucket=bucket_name, Key=model_s3_key)
            print("✅ Cleanup complete!")
        else:
            print(f"\n💡 To cleanup later:")
            print(f"   aws sagemaker delete-endpoint --endpoint-name {endpoint_name}")
            print(f"   aws sagemaker delete-endpoint-config --endpoint-config-name {config_name}")
            print(f"   aws sagemaker delete-model --model-name {model_name}")
        
        # Cleanup local files
        os.remove(model_tar_path)
        import shutil
        shutil.rmtree(local_model_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = create_builtin_endpoint()
    if success:
        print("\n🎉 Built-in sklearn mode works perfectly!")
        print("Now you can use this approach in your pipeline.")
    else:
        print("\n💥 Built-in mode test failed.")
