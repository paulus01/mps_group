import boto3
import os

def handler(event, context):
    region = os.environ["AWS_REGION"]
    sm = boto3.client('sagemaker', region_name=region)
    resp = sm.start_pipeline_execution(
        PipelineName=os.environ['PIPELINE_NAME'],
        PipelineParameters=[
            {
                'Name': 'InputDataUrl',
                'Value': os.environ['INPUT_DATA_URL']
            }
        ]
    )
    return {'PipelineExecutionArn': resp['PipelineExecutionArn']}