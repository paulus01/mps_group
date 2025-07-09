import boto3
import yaml

# Load config
with open("config/test.yaml", "r") as f:
    config = yaml.safe_load(f)

pipeline_name = "IrisPipelineV2"  # Must match the name in your CDK pipeline definition
region = config["aws"]["region"]

# Optional: Pass pipeline parameters from config if needed
pipeline_parameters = [
    {
        "Name": "InputDataUrl",
        "Value": f"s3://{config['bucket_name']}/{config['dataset_key']}"
    }
]

sagemaker = boto3.client("sagemaker", region_name=region)

response = sagemaker.start_pipeline_execution(
    PipelineName=pipeline_name,
    PipelineParameters=pipeline_parameters
)

print("Started pipeline execution:", response["PipelineExecutionArn"])