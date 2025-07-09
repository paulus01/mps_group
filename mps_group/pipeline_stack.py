from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_lambda as _lambda,
    custom_resources as cr,
    Duration,
    aws_s3_assets as s3_assets,
    aws_events as events,
    aws_events_targets as targets,
)
from constructs import Construct
import json
import sagemaker as sm_sdk
import os


class PipelineStack(Stack):
    def __init__(self, scope: Construct, id: str, config: dict, vpc, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Configuration
        bucket_name = config.get('bucket_name', 'default-bucket-name')
        dataset_key = config.get('dataset_key', 'default-dataset-key')
        region = config.get('aws', {}).get('region', 'us-east-2')
        
        # XGBoost container image
        image_uri = sm_sdk.image_uris.retrieve(
            framework="xgboost",
            region=region,
            version="1.7-1",
            instance_type="ml.m5.xlarge"
        )

        # SageMaker Domain configuration
        domain_cfg = config.get('sagemaker_domain', {})
        domain_name = domain_cfg.get('domain_name', 'iris-domain')
        user_profile_name = domain_cfg.get('user_profile_name', 'default-user')
        subnet_ids = [subnet.subnet_id for subnet in vpc.private_subnets]

        # IAM Execution Role
        execution_role = iam.Role(
            self, "SageMakerPipelineExecutionRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
                iam.ServicePrincipal("lambda.amazonaws.com")
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly")
            ],
            inline_policies={
                "ECRAccessPolicy": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "ecr:BatchCheckLayerAvailability",
                                "ecr:GetDownloadUrlForLayer",
                                "ecr:BatchGetImage",
                                "ecr:GetAuthorizationToken",
                                "ecr:DescribeRepositories",
                                "ecr:ListImages",
                                "ecr:DescribeImages"
                            ],
                            resources=["*"]
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "ecr:BatchCheckLayerAvailability",
                                "ecr:GetDownloadUrlForLayer", 
                                "ecr:BatchGetImage"
                            ],
                            resources=[f"arn:aws:ecr:{region}:683313688378:repository/*"]
                        )
                    ]
                )
            }
        )

        # SageMaker Domain and User Profile
        domain = sagemaker.CfnDomain(
            self, "SageMakerDomain",
            auth_mode="IAM",
            domain_name=domain_name,
            subnet_ids=subnet_ids,
            vpc_id=vpc.vpc_id,
            default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                execution_role=execution_role.role_arn
            ),
            app_network_access_type="VpcOnly"
        )

        user_profile = sagemaker.CfnUserProfile(
            self, "SageMakerUserProfile",
            domain_id=domain.attr_domain_id,
            user_profile_name=user_profile_name,
            user_settings=sagemaker.CfnUserProfile.UserSettingsProperty(
                execution_role=execution_role.role_arn
            )
        )

        # === PREPROCESSING STEP ===
        preprocessing_asset = s3_assets.Asset(
            self, "PreprocessingScriptAsset",
            path="mps_group/processing/preprocessing.py"
        )
        preprocessing_asset.grant_read(execution_role)

        preprocessing_step = {
            "Name": "IrisPreprocessing",
            "Type": "Processing",
            "Arguments": {
                "ProcessingJobName": "iris-preprocessing-job",
                "ProcessingResources": {
                    "ClusterConfig": {
                        "InstanceCount": 1,
                        "InstanceType": "ml.m5.xlarge",
                        "VolumeSizeInGB": 30
                    }
                },
                "AppSpecification": {
                    "ImageUri": image_uri,
                    "ContainerEntrypoint": [
                        "python3",
                        f"/opt/ml/processing/input/code/{os.path.basename(preprocessing_asset.s3_object_key)}"
                    ]
                },
                "RoleArn": execution_role.role_arn,
                "ProcessingInputs": [
                    {
                        "InputName": "input-1",
                        "S3Input": {
                            "S3Uri": {"Get": "Parameters.InputDataUrl"},
                            "LocalPath": "/opt/ml/processing/input/data",
                            "S3DataType": "S3Prefix",
                            "S3InputMode": "File"
                        }
                    },
                    {
                        "InputName": "code",
                        "S3Input": {
                            "S3Uri": f"s3://{preprocessing_asset.s3_bucket_name}/{preprocessing_asset.s3_object_key}",
                            "LocalPath": "/opt/ml/processing/input/code",
                            "S3DataType": "S3Prefix",
                            "S3InputMode": "File"
                        }
                    }
                ],
                "ProcessingOutputConfig": {
                    "Outputs": [
                        {
                            "OutputName": "output-1",
                            "S3Output": {
                                "S3Uri": f"s3://{bucket_name}/iris/output",
                                "LocalPath": "/opt/ml/processing/output",
                                "S3UploadMode": "EndOfJob"
                            }
                        }
                    ]
                }
            }
        }

        # === TRAINING STEP ===
        training_asset = s3_assets.Asset(
            self, "TrainingScriptAsset",
            path="mps_group/training/train_xgboost.py"
        )
        training_asset.grant_read(execution_role)

        training_step = {
            "Name": "IrisTraining",
            "Type": "Training",
            "DependsOn": ["IrisPreprocessing"],
            "Arguments": {
                "TrainingJobName": "iris-training-job",
                "AlgorithmSpecification": {
                    "TrainingImage": image_uri,
                    "TrainingInputMode": "File",
                    "ContainerEntrypoint": [
                        "python3",
                        f"/opt/ml/input/data/code/{os.path.basename(training_asset.s3_object_key)}"
                    ]
                },
                "RoleArn": execution_role.role_arn,
                "InputDataConfig": [
                    {
                        "ChannelName": "code",
                        "DataSource": {
                            "S3DataSource": {
                                "S3Uri": f"s3://{training_asset.s3_bucket_name}/{training_asset.s3_object_key}",
                                "S3DataType": "S3Prefix"
                            }
                        },
                        "InputMode": "File"
                    },
                    {
                        "ChannelName": "train",
                        "DataSource": {
                            "S3DataSource": {
                                "S3Uri": {
                                    "Get": "Steps.IrisPreprocessing.ProcessingOutputConfig.Outputs['output-1'].S3Output.S3Uri"
                                },
                                "S3DataType": "S3Prefix"
                            }
                        },
                        "ContentType": "text/csv",
                        "InputMode": "File"
                    }
                ],
                "OutputDataConfig": {
                    "S3OutputPath": f"s3://{bucket_name}/iris/model"
                },
                "ResourceConfig": {
                    "InstanceType": "ml.m5.xlarge",
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 30
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 3600
                }
            }
        }

        # === MODEL REGISTRY STEP ===
        model_package_group_name = "IrisModelPackageGroup"
        sagemaker.CfnModelPackageGroup(
            self, "IrisModelPackageGroup",
            model_package_group_name=model_package_group_name,
            model_package_group_description="Model group for Iris classification models"
        )

        register_model_step = {
            "Name": "RegisterTrainedModel",
            "Type": "RegisterModel",
            "Arguments": {
                "ModelPackageGroupName": model_package_group_name,
                "ModelApprovalStatus": "Approved",
                "InferenceSpecification": {
                    "Containers": [
                        {
                            "Image": image_uri,
                            "ModelDataUrl": {
                                "Get": "Steps.IrisTraining.ModelArtifacts.S3ModelArtifacts"
                            }
                        }
                    ],
                    "SupportedContentTypes": ["text/csv"],
                    "SupportedResponseMIMETypes": ["text/csv"],
                    "SupportedRealtimeInferenceInstanceTypes": ["ml.t2.medium"],
                    "SupportedTransformInstanceTypes": ["ml.m5.large"]
                }
            }
        }

        # === SAGEMAKER PIPELINE ===
        pipeline_definition = {
            "Version": "2020-12-01",
            "Parameters": [
                {
                    "Name": "InputDataUrl",
                    "Type": "String",
                    "DefaultValue": f"s3://{bucket_name}/{dataset_key}"
                }
            ],
            "Steps": [
                preprocessing_step,
                training_step,
                register_model_step
            ]
        }

        sagemaker.CfnPipeline(
            self, "IrisPipeline",
            pipeline_name="IrisPipelineV2",
            pipeline_definition={
                "PipelineDefinitionBody": json.dumps(pipeline_definition)
            },
            role_arn=execution_role.role_arn
        )

        # === PIPELINE TRIGGER LAMBDA ===
        trigger_lambda = _lambda.Function(
            self, "TriggerPipelineLambda",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="trigger_pipeline_preprocessing.handler",
            code=_lambda.Code.from_asset("mps_group/lambda/trigger_pipeline_preprocessing/"),
            timeout=Duration.seconds(60),
            environment={
                "PIPELINE_NAME": "IrisPipelineV2",
                "INPUT_DATA_URL": f"s3://{bucket_name}/{dataset_key}"
            }
        )
        trigger_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["sagemaker:StartPipelineExecution"],
                resources=["*"]
            )
        )

        # === MODEL DEPLOYMENT LAMBDA ===
        deploy_lambda = _lambda.Function(
            self, "DeployModelLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="deploy_model.handler",
            code=_lambda.Code.from_asset("mps_group/lambda/deploy_model"),
            timeout=Duration.seconds(300),
            environment={
                "MODEL_PACKAGE_GROUP_NAME": model_package_group_name,
                "ENDPOINT_NAME": "iris-classification-endpoint",
                "ENDPOINT_CONFIG_NAME": "iris-classification-config",
                "SAGEMAKER_EXECUTION_ROLE_ARN": execution_role.role_arn,
                "SAGEMAKER_FRAMEWORK": "xgboost"
            }
        )
        deploy_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:ListModelPackages",
                    "sagemaker:DescribeModelPackage",
                    "sagemaker:CreateModel",
                    "sagemaker:CreateEndpointConfig",
                    "sagemaker:CreateEndpoint",
                    "sagemaker:DescribeEndpoint",
                    "sagemaker:DescribeEndpointConfig",
                    "sagemaker:DescribeModel",
                    "sagemaker:UpdateEndpoint",
                    "sagemaker:DeleteEndpoint",
                    "sagemaker:DeleteEndpointConfig",
                    "sagemaker:DeleteModel",
                    "iam:PassRole"
                ],
                resources=["*"]
            )
        )

        # === EVENT TRIGGERS ===
        # Trigger pipeline on stack creation
        cr.AwsCustomResource(
            self, "TriggerPipelineCustomResource",
            on_create=cr.AwsSdkCall(
                service="Lambda",
                action="invoke",
                parameters={
                    "FunctionName": trigger_lambda.function_name,
                    "InvocationType": "Event"
                },
                physical_resource_id=cr.PhysicalResourceId.of("TriggerPipelineCustomResource")
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements([
                iam.PolicyStatement(
                    actions=["lambda:InvokeFunction"],
                    resources=[trigger_lambda.function_arn]
                )
            ])
        )

        # Deploy model when registered
        deployment_rule = events.Rule(
            self, "ModelRegistrationCompletionRule",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Model Package State Change"],
                detail={
                    "ModelPackageGroupName": [model_package_group_name],
                    "ModelApprovalStatus": ["Approved"]
                }
            )
        )
        deployment_rule.add_target(targets.LambdaFunction(deploy_lambda))

        # Deploy latest model immediately
        cr.AwsCustomResource(
            self, "DeployLatestModelResource",
            on_create=cr.AwsSdkCall(
                service="Lambda",
                action="invoke",
                parameters={
                    "FunctionName": deploy_lambda.function_name,
                    "InvocationType": "Event"
                },
                physical_resource_id=cr.PhysicalResourceId.of("DeployLatestModelResource")
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements([
                iam.PolicyStatement(
                    actions=["lambda:InvokeFunction"],
                    resources=[deploy_lambda.function_arn]
                )
            ])
        )