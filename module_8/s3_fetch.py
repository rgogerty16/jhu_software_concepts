"""boto3 helpers to move the Grad Cafe dataset between S3 and SageMaker.

This extends the Module 7 workflow. Credentials are resolved through boto3's
default provider chain -- on a SageMaker notebook instance that means the
attached IAM execution role, and locally it means the usual environment /
shared-config credentials. Keys are therefore never hard-coded here or in the
notebook that imports this module.

For Module 8 an ``upload_dataset`` helper is added so the cleaned dataset can be
written back to the same bucket (assignment step 9).

Offline development: if the environment variable ``LOCAL_DATASET`` points at a
JSON file, ``download_dataset`` copies that file instead of calling S3, and
``upload_dataset`` prints a simulated confirmation. On SageMaker that variable is
unset, so the real S3 calls run. This keeps a single notebook runnable both on
the laptop (for authoring) and on SageMaker (for the graded run).
"""

import os
import shutil

import boto3

# Defaults are overridable via environment variables (see ``.env.example``) so
# the same code works whether the bucket is ``grad-cafe`` or ``grad-cafe-<init>``.
DEFAULT_BUCKET = os.getenv("S3_BUCKET", "grad-cafe-rg")
DEFAULT_KEY = os.getenv("S3_KEY", "applicant_data.json")
DEFAULT_OUTPUT = os.getenv("OUTPUT_FILE", "applicant_data_SM.json")
DEFAULT_CLEAN_KEY = os.getenv("S3_CLEAN_KEY", "cleaned_gradcafe.json")


def make_s3_client(region=None):
    """Return an S3 client whose credentials come from boto3's default chain.

    ``region`` falls back to the ``AWS_REGION`` environment variable and then to
    the instance/notebook default, so no region need be passed on SageMaker.
    """
    session = boto3.session.Session(region_name=region or os.getenv("AWS_REGION"))
    return session.client("s3")


def download_dataset(bucket=DEFAULT_BUCKET, key=DEFAULT_KEY,
                     output_path=DEFAULT_OUTPUT, client=None):
    """Download ``key`` from S3 ``bucket`` and save it locally as ``output_path``.

    A client is created from the default credential chain when one is not
    supplied. Returns the path written, so callers can print or reuse it.
    """
    local = os.getenv("LOCAL_DATASET")
    if local:
        # Offline authoring mode: stand in for S3 with a local copy.
        shutil.copyfile(local, output_path)
        return output_path
    client = client or make_s3_client()
    client.download_file(bucket, key, output_path)
    return output_path


def upload_dataset(local_path, bucket=DEFAULT_BUCKET, key=DEFAULT_CLEAN_KEY,
                   client=None):
    """Upload ``local_path`` back to S3 ``bucket`` under ``key``.

    Returns the ``s3://bucket/key`` URI written. In offline authoring mode
    (``LOCAL_DATASET`` set) the real upload is skipped and a simulated URI is
    returned so the notebook still runs end to end without AWS credentials.
    """
    uri = f"s3://{bucket}/{key}"
    if os.getenv("LOCAL_DATASET"):
        return uri
    client = client or make_s3_client()
    client.upload_file(local_path, bucket, key)
    return uri
