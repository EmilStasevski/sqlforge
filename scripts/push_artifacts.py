"""Upload model + evaluation artifacts to MinIO (S3 API): s3://sqlforge-artifacts/<git-sha>/...

Datasets go through DVC (`make data-push`); this script handles the adapter, GGUF, eval reports and
predictions. Credentials default to the compose MinIO (override with AWS_* / S3_ENDPOINT env vars).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import boto3

BUCKET = "sqlforge-artifacts"


def main() -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT", "http://localhost:9002"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "sqlforge"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "sqlforge-secret"),
        region_name="us-east-1",
    )
    existing = {b["Name"] for b in s3.list_buckets()["Buckets"]}
    if BUCKET not in existing:
        s3.create_bucket(Bucket=BUCKET)
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    files: list[Path] = []
    for pattern in ("runs/qlora/adapter/*", "runs/serve/*", "results/*.json", "data/eval/*.parquet"):
        files += [p for p in Path().glob(pattern) if p.is_file()]
    manifest = []
    for p in files:
        key = f"{sha}/{p.as_posix()}"
        s3.upload_file(str(p), BUCKET, key)
        manifest.append({"key": key, "bytes": p.stat().st_size})
        print("uploaded", key)
    s3.put_object(Bucket=BUCKET, Key=f"{sha}/manifest.json", Body=json.dumps(manifest, indent=2).encode())
    print(f"{len(manifest)} files -> s3://{BUCKET}/{sha}/")


if __name__ == "__main__":
    main()
