import subprocess
import glob
import shutil
import tarfile
import os
import numpy as np
import logging
from fire import Fire
from time import time, sleep
import multiprocessing
from tqdm import tqdm

os.environ["AWS_DEFAULT_REGION"] = "us-west-1"
os.environ["AWS_ACCESS_KEY_ID"] = "AKIAZW3TMCLLUA6MAQLI"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"

from typing import List, Optional, Literal
from moto3.queue_manager import QueueManager
from google.cloud import storage, compute_v1
from itertools import repeat
import traceback


IDENTIFIERS = [
    "unavailable",
    "private",
    "terminated",
    "removed",
    "country",
    "closed",
    "copyright",
    "members",
    "not available",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("download.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def download_to_vm_from_gcs(bucket_name, gcs_file_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # Initialize a storage client
    storage_client = storage.Client()
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    # Get the blob (file in GCS)
    blob = bucket.blob(gcs_file_name)
    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    # Download the blob to a local file
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {gcs_file_name} from bucket {bucket_name} to {destination_file_name}.")


def upload_tar_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads data to a GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logger.info(f"File {destination_blob_name} uploaded to {bucket_name}.")
    

def compress_dir(source_dir, tar_file):
    with tarfile.open(tar_file, "w:gz") as tar:
        tar.add(source_dir, arcname=".")


def shutdown_vm(project, zone, instance_name):
    client = compute_v1.InstancesClient()

    # Request to stop the instance
    operation = client.delete(project=project, zone=zone, instance=instance_name)

    # Wait for the operation to complete
    operation.result()
    

def download(data_dir, metadata_path) -> Optional[None]:        
    """
    -- python download_upstream.py \
    --scale datacomp_1b \
    --data_dir /weka/oe_data_default/georges/data/datacomp-1b/shards/$SHARD/ \
    --metadata_dir /weka/oe_data_default/jamesp/data/datacomp-1b/metadata/$SHARD/ \
    --enable_wandb \
    --wandb_project datacompShardTest \
    --retries 5 \
    --resize_mode no \
    --skip_bbox_blurring
    """
    
    hash_name = data_dir.split('/')[-1]
    script = "download_upstream.py"
    
    args = [
        "--scale",
        "xlarge",
        "--data_dir",
        data_dir,
        "--metadata_dir",
        os.path.dirname(metadata_path),
        "--retries",
        "5",
        "--resize_mode",
        "no",
        "--skip_bbox_blurring",
        "--download_npz",
        "--skip_shards"
    ]
            
    try:
        result = subprocess.run(
                ["python", script] + args, 
                capture_output=True, 
                text=True
            )

        if any(identifier in result.stderr.lower() for identifier in IDENTIFIERS):
            with open(f"metadata/unavailable/{data_dir}.txt", "a") as f:
                f.write(f"{hash_name}\n")
            return "unavailable"
        elif "bot" in result.stderr.lower():
            with open(f"metadata/blocked_ip/{data_dir}.txt", "a") as f:
                f.write(f"{hash_name}\t{result.stderr}\t{result.stdout}\n")
            return "blocked IP"
        else:
            if not os.path.exists(f"{data_dir}/{hash_name}/{hash_name}.parquet"):
                with open(f"metadata/unknown/{data_dir}.txt", "a") as f:
                    f.write(f"{hash_name}\taudio\t{result.stderr}\n")

            # if not os.path.exists(
            #     f"{data_dir}/{hash_name}/{hash_name}.{lang_code}.{sub_format}"
            # ):
            #     with open(f"metadata/unknown/{data_dir}.txt", "a") as f:
            #         f.write(f"{hash_name}\ttranscript\t{result.stderr}\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in downloading video {hash_name}: {e.stderr}")
        return -1


def parallel_download(arguments):
    return download(*arguments)


def get_vm_info():
    cmd = [
        "curl",
        "http://metadata.google.internal/computeMetadata/v1/instance/name",
        "-H",
        "Metadata-Flavor: Google",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    instance_name = result.stdout.strip()

    cmd = [
        "curl",
        "http://metadata.google.internal/computeMetadata/v1/instance/zone",
        "-H",
        "Metadata-Flavor: Google",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    zone = result.stdout.strip().split("/")[-1]

    return instance_name, zone


def upload_dir_to_gcs(bucket_name, source_dir):
    """Uploads data to a GCS bucket."""
    content = glob.glob(f"{source_dir}/*/*")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    for file in content:
        blob = bucket.blob(file)
        blob.upload_from_filename(file)
        logger.info(f"File {file} uploaded to {bucket_name}.")
    return content


def download_to_gcs(data_dir, metadata_path) -> None:
    logger.info(f"Processing into: {data_dir} from: {metadata_path}..")

    start = time()
    
    with multiprocessing.Pool() as pool:
        res = list(
            tqdm(
                pool.imap_unordered(
                    parallel_download,
                    zip(
                        repeat(data_dir),
                        repeat(metadata_path)
                    ),
                ),
                total=1, # we just call the script once
            )
        )
    duration = (time() - start) / 60
    logger.info(f"Downloaded one shard ({metadata_path}) in {duration:.2f} minutes.")

    if duration < 3 or "blocked IP" in res:
        logger.error(f"Blocked IP detected.")
        with open(f"metadata/blocked_ip/{data_dir}.txt", "a") as f:
            f.write(f"DEFINITELY_REDOWNLOAD\n")
        return "blocked IP"
    

def main(bucket, queue_id):
    instance_name, zone = get_vm_info()
    logger.info(f"Instance name: {instance_name}..\n")
    qm = QueueManager(queue_id)
    os.makedirs("metadata", exist_ok=True)
    os.makedirs("metadata/unavailable", exist_ok=True)
    os.makedirs("metadata/blocked_ip", exist_ok=True)
    os.makedirs("metadata/unknown", exist_ok=True)

    while True:
        try:
            message, item = qm.get_next(
                visibility_timeout=(180 * 60)
            )  # 1 hour visibility timeout

            metadata_path = item["metadata_path"] # this dir actually a path
            data_dir = os.path.join(os.getcwd(), item["data_dir"]) # this is actually a dir.
            bucket_name = item["bucket_name"]

            logger.info("Downloading audio and transcript files")
            
            download_to_vm_from_gcs(
                bucket_name=bucket_name,
                gcs_file_name=metadata_path,
                destination_file_name=os.path.join(os.getcwd(), metadata_path),
            )
            
            status = download_to_gcs(
                data_dir=data_dir,
                metadata_path=metadata_path
            )
            
            # qm.delete(message)

            if status == "blocked IP":
                logger.info("Blocked IP detected. Requeuing item.")
                logger.info("Uploading metadata to GCS..")
                # Why does this need to happen? Isn't the data already in GCS?
                # metadata_files = upload_metadata_to_gcs(
                #     bucket_name=bucket,
                #     source_dir="metadata",
                # )
                qm.delete(message)
                qm.upload([item])
                break
            else:
                logger.info("Compressing files..")
                start = time()
                # compress_dir(data_dir, f"{data_dir}.tar.gz")
                dur = (time() - start) / 60
                logger.info(f"Compressed files in {dur:.2f} minutes.")

                # logger.info(f"Uploading {data_dir}.tar.gz to GCS..")
                # upload_tar_to_gcs(
                #     bucket_name=bucket,
                #     source_file_name=f"{data_dir}.tar.gz",
                #     destination_blob_name=f"ow_1M_full/{output_dir}.tar.gz",
                # )

                logger.info(f"Uploading metadata to GCS..")
                metadata_files = upload_dir_to_gcs(
                    bucket_name=bucket,
                    source_dir=data_dir,
                )

                shutil.rmtree(data_dir)
                # os.remove(f"{output_dir}.tar.gz")
                qm.delete(message)
                
            for file in metadata_files:
                os.remove(file)
        
        
        except IndexError:  # no more items in queue to process
            logger.info("No more items in queue to process")
            shutdown_vm("oe-training", zone, instance_name)
            break
        except Exception:  # error occurred
            logger.info("An error occurred. Requeuing item")
            logger.info(traceback.format_exc())
            qm.delete(message)
            qm.upload([item])

    if status == "blocked IP":
        shutdown_vm("oe-training", zone, instance_name)