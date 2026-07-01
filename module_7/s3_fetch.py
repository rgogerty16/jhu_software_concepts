"""boto3 helpers to fetch the Grad Cafe dataset from S3 into SageMaker.

Credentials are resolved through boto3's default provider chain — on a SageMaker
notebook instance that means the attached IAM execution role, and locally it
means the usual environment / shared-config credentials. Keys are therefore
never hard-coded in this module or in the notebook that imports it.

The notebook (``grad-cafe-pipeline.ipynb``) does nothing more than call these
functions, which keeps the notebook itself trivially lint-clean.
"""

import json
import os

import boto3

# Defaults are overridable via environment variables (see ``.env.example``) so
# the same code works whether the bucket is ``grad-cafe`` or ``grad-cafe-<init>``.
DEFAULT_BUCKET = os.getenv("S3_BUCKET", "grad-cafe")
DEFAULT_KEY = os.getenv("S3_KEY", "applicant_data.json")
DEFAULT_OUTPUT = os.getenv("OUTPUT_FILE", "applicant_data_SM.json")


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
    client = client or make_s3_client()
    client.download_file(bucket, key, output_path)
    return output_path


def preview_dataset(path=DEFAULT_OUTPUT):
    """Load a downloaded JSON dataset and return a short summary dictionary.

    Used by the notebook to prove the download succeeded: it reports the record
    count and the fields present on the first record.
    """
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    summary = {"path": path, "record_count": len(records)}
    if records:
        summary["first_record_keys"] = sorted(records[0].keys())
    return summary
