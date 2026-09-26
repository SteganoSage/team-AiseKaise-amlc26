"""Submit the student_resource BER pipeline as a SageMaker Processing job.

Example from a SageMaker notebook or an AWS-configured local environment:

    python submit_sagemaker_job.py \
        --dataset-s3 s3://my-bucket/student_resource/dataset \
        --output-s3 s3://my-bucket/student_resource/sagemaker-output \
        --role arn:aws:iam::123456789012:role/SageMakerExecutionRole

The dataset S3 URI must point to the directory containing ``train/`` and
``test/``. The job writes matching results, candidate pairs, run metadata,
the model, and logs under the output prefix.
"""

import argparse
import os

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit BER to SageMaker Processing")
    parser.add_argument("--dataset-s3", required=True, help="S3 prefix containing train/ and test/")
    parser.add_argument("--output-s3", required=True, help="S3 prefix for job outputs")
    parser.add_argument("--role", default=None, help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", default="ml.m5.4xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--mode", choices=("validate", "loco", "test"), default="test")
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Wait for the job to finish")
    args = parser.parse_args()

    session = sagemaker.Session(boto_session=boto3.Session())
    role = args.role or sagemaker.get_execution_role()
    region = session.boto_region_name
    repo_dir = os.path.dirname(os.path.abspath(__file__))

    processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.2-1",
            py_version="py3",
            instance_type=args.instance_type,
        ),
        role=role,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        command=["python3"],
        base_job_name="student-resource-ber",
        sagemaker_session=session,
    )

    arguments = ["--mode", args.mode]
    if args.embeddings:
        arguments.append("--embeddings")
    if args.skip_install:
        arguments.append("--skip-install")

    processor.run(
        code="sagemaker_process.py",
        source_dir=repo_dir,
        inputs=[
            ProcessingInput(
                source=args.dataset_s3.rstrip("/"),
                destination="/opt/ml/processing/input/dataset",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="results",
                source="/opt/ml/processing/output",
                destination=args.output_s3.rstrip("/") + "/output",
            ),
            ProcessingOutput(
                output_name="model",
                source="/opt/ml/processing/model",
                destination=args.output_s3.rstrip("/") + "/model",
            ),
        ],
        arguments=arguments,
        wait=args.wait,
    )

    print(f"Submitted SageMaker Processing job. Region: {region}")
    print(f"Outputs: {args.output_s3.rstrip('/')}/output and /model")


if __name__ == "__main__":
    main()