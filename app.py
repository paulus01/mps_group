#!/usr/bin/env python3
import aws_cdk as cdk
import yaml
import os

from mps_group.s3_stack import S3Stack
from mps_group.vpc_stack import VpcStack
from mps_group.pipeline_stack import PipelineStack

with open("config/test.yaml") as f:
    config = yaml.safe_load(f)

account = config["aws"]["account"] = os.environ.get("account", config["aws"]["account"])
region = config["aws"]["region"]

app = cdk.App()
env = cdk.Environment(account=account, region=region)

S3Stack(
    app,
    "MpsGroupS3Stack",
    config=config,
    env=env
)

vpc_stack = VpcStack(app, "MpsGroupVpcStack", env=env)

PipelineStack(
    app,
    "MpsGroupPipelineStack",
    config=config,
    vpc=vpc_stack.vpc,   # Pass the VPC object
    env=env
)

app.synth()
