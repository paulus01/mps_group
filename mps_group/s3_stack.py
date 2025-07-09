"""
S3 Stack for ML Pipeline Data Storage

Creates an S3 bucket and automatically uploads the Iris dataset.
"""

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CustomResource,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_logs as logs,
    aws_iam as iam,
)
from aws_cdk.custom_resources import Provider
from constructs import Construct


class S3Stack(Stack):
    """CDK Stack for S3 bucket and dataset upload"""

    def __init__(self, scope: Construct, construct_id: str, config: dict, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # Configuration
        bucket_name = config["bucket_name"]
        dataset_url = config["dataset_url"]
        dataset_key = config["dataset_key"]
        removal_policy = self._get_removal_policy(config)

        # Create S3 bucket
        self.bucket = s3.Bucket(
            self,
            "DatasetBucket",
            bucket_name=bucket_name,
            versioned=config.get("versioned", False),
            removal_policy=removal_policy,
            auto_delete_objects=(removal_policy == RemovalPolicy.DESTROY),
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # Create Lambda for dataset upload
        uploader_lambda = _lambda.Function(
            self,
            "UploadDatasetFunction",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="upload_dataset.handler",
            code=_lambda.Code.from_asset("mps_group/lambda/upload_dataset"),
            timeout=Duration.seconds(300),
            environment={
                "DATASET_URL": dataset_url,
                "BUCKET_NAME": self.bucket.bucket_name,
                "DATASET_KEY": dataset_key,
            },
        )

        # Configure permissions
        self.bucket.grant_put(uploader_lambda)
        uploader_lambda.role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSLambdaBasicExecutionRole"
            )
        )

        # Create custom resource to trigger upload
        provider = Provider(
            self,
            "DatasetUploadProvider",
            on_event_handler=uploader_lambda,
            log_retention=logs.RetentionDays.ONE_WEEK,
        )

        CustomResource(
            self,
            "TriggerDatasetUpload",
            service_token=provider.service_token,
        )

    def _get_removal_policy(self, config: dict) -> RemovalPolicy:
        """Get removal policy from config"""
        policy_str = config.get("removal_policy", "RETAIN").upper()
        return (
            RemovalPolicy.DESTROY
            if policy_str == "DESTROY"
            else RemovalPolicy.RETAIN
        )

    @property
    def bucket_name(self) -> str:
        """Get the bucket name"""
        return self.bucket.bucket_name

    @property
    def bucket_arn(self) -> str:
        """Get the bucket ARN"""
        return self.bucket.bucket_arn
