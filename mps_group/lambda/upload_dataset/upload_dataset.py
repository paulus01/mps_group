import json
import boto3
import urllib.request
import urllib.error
import ssl
import os

s3 = boto3.client("s3")

def handler(event, context):
    print("Received event:", json.dumps(event))
    request_type = event.get("RequestType")
    if request_type == "Delete":
        return {"Status": "SUCCESS"}

    try:
        dataset_url = os.environ["DATASET_URL"]
        bucket_name = os.environ["BUCKET_NAME"]
        object_key = os.environ["DATASET_KEY"]

        print(f"Downloading dataset from {dataset_url}...")
        
        # Create SSL context that doesn't verify certificates (for simplicity)
        # In production, you might want to handle certificates properly
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Use urllib instead of requests
        with urllib.request.urlopen(dataset_url, context=ssl_context) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}: Failed to download dataset")
            
            dataset_content = response.read()

        print(f"Uploading to s3://{bucket_name}/{object_key}...")
        s3.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=dataset_content,
            ContentType="text/csv"
        )

        return {
            "Status": "SUCCESS",
            "PhysicalResourceId": object_key,
            "Data": {
                "Message": f"Dataset uploaded to s3://{bucket_name}/{object_key}"
            }
        }

    except Exception as e:
        print("Error occurred:", str(e))
        return {
            "Status": "FAILED",
            "Reason": str(e),
            "PhysicalResourceId": "upload-failed"
        }
