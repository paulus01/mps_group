#!/usr/bin/env python3
"""
Test script to verify the deployed SageMaker inference endpoint.
This script sends sample iris data to the endpoint and verifies the response.
"""

import boto3
import json
import numpy as np

def test_endpoint():
    """Test the deployed iris classification endpoint"""
    
    # Initialize SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime', region_name='us-east-2')
    
    endpoint_name = 'iris-classification-endpoint'
    
    print(f"Testing endpoint: {endpoint_name}")
    
    # Sample iris data (sepal_length, sepal_width, petal_length, petal_width)
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],  # Should be class 0 (setosa)
        [7.0, 3.2, 4.7, 1.4],  # Should be class 1 (versicolor)
        [6.3, 3.3, 6.0, 2.5],  # Should be class 2 (virginica)
    ]
    
    class_names = ['setosa', 'versicolor', 'virginica']
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\n--- Test Sample {i} ---")
        print(f"Input: {sample}")
        
        try:
            # Convert sample to CSV format
            csv_input = ','.join(map(str, sample))
            
            # Make prediction
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='text/csv',
                Accept='text/csv',
                Body=csv_input
            )
            
            # Parse response
            result = response['Body'].read().decode()
            print(f"Raw response: {result}")
            
            # Try to parse as prediction
            try:
                prediction = int(float(result.strip()))
                predicted_class = class_names[prediction] if prediction < len(class_names) else f"Unknown class {prediction}"
                print(f"Predicted class: {prediction} ({predicted_class})")
            except ValueError:
                print(f"Could not parse prediction: {result}")
                
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
    
    print("\nEndpoint testing completed!")

def check_endpoint_status():
    """Check the status of the endpoint"""
    
    sagemaker = boto3.client('sagemaker', region_name='us-east-2')
    endpoint_name = 'iris-classification-endpoint'
    
    try:
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        
        print(f"Endpoint Name: {response['EndpointName']}")
        print(f"Endpoint Status: {response['EndpointStatus']}")
        print(f"Endpoint ARN: {response['EndpointArn']}")
        print(f"Creation Time: {response['CreationTime']}")
        
        if response['EndpointStatus'] == 'InService':
            print("Endpoint is ready for inference!")
            return True
        elif response['EndpointStatus'] == 'Creating':
            print("Endpoint is still being created...")
            return False
        else:
            print(f"Endpoint status: {response['EndpointStatus']}")
            if 'FailureReason' in response:
                print(f"Failure reason: {response['FailureReason']}")
            return False
            
    except Exception as e:
        print(f"Error checking endpoint status: {str(e)}")
        return False

if __name__ == "__main__":
    print("Checking endpoint status...")
    if check_endpoint_status():
        print("\nTesting endpoint with sample data...")
        test_endpoint()
    else:
        print("\nEndpoint is not ready for testing yet. Please wait and try again.")
        print("\nTo check endpoint status, you can also run:")
        print("aws sagemaker describe-endpoint --endpoint-name iris-classification-endpoint")
