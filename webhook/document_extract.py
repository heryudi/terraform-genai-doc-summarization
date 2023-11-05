# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
import uuid
from google.cloud import vision
from google.cloud import storage


def clear_ocr_output_folder(bucket_name: str, prefix: str) -> None:
    """Deletes all blobs with the given prefix in the specified bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        blob.delete()
    print(f"All blobs with prefix '{prefix}' have been deleted from bucket '{bucket_name}'.")


def async_document_extract(
    bucket: str,
    name: str,
    output_bucket: str,
    timeout: int = 420,
) -> str:
    """Perform OCR with PDF/TIFF as source files on GCS.

    Original sample is here:
    https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/vision/snippets/detect/detect.py#L806

    Note: This function can cause the IOPub data rate to be exceeded on a
    Jupyter server. This rate can be changed by setting the variable
    `--ServerApp.iopub_data_rate_limit

    Args:
        bucket (str): GCS URI of the bucket containing the PDF/TIFF files.
        name (str): name of the PDF/TIFF file.
        output_bucket: bucket to store output in
        timeout (int): Timeout in seconds for the request.


    Returns:
        str: the complete text
    """
    # Generate a unique identifier for this OCR job
    unique_id = uuid.uuid4().hex
    prefix = f"ocr/{unique_id}"

    # Before starting a new OCR job, clear the output folder of previous job results with the same unique ID
    clear_ocr_output_folder(output_bucket, prefix)

    gcs_source_uri = f"gs://{bucket}/{name}"
    gcs_destination_uri = f"gs://{output_bucket}/{prefix}/"
    mime_type = "application/pdf"
    batch_size = 2

    # Perform Vision OCR
    client = vision.ImageAnnotatorClient()

    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

    gcs_source = vision.GcsSource(uri=gcs_source_uri)
    input_config = vision.InputConfig(gcs_source=gcs_source, mime_type=mime_type)

    gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
    output_config = vision.OutputConfig(
        gcs_destination=gcs_destination, batch_size=batch_size
    )

    async_request = vision.AsyncAnnotateFileRequest(
        features=[feature], input_config=input_config, output_config=output_config
    )

    operation = client.async_batch_annotate_files(requests=[async_request])

    print("OCR: waiting for the operation to finish.")
    operation.result(timeout=timeout)

    # Once the request has completed and the output has been
    # written to GCS, we can list all the output files.
    return get_ocr_output_from_bucket(gcs_destination_uri, output_bucket)


def get_ocr_output_from_bucket(gcs_destination_uri: str, bucket_name: str) -> str:
    """Iterates over blobs in output bucket to get full OCR result in the correct order.

    Arguments:
        gcs_destination_uri: the URI where the OCR output was saved.
        bucket_name: the name of the bucket where the output was saved.

    Returns:
        The full text of the document in the correct order
    """
    storage_client = storage.Client()
    match = re.match(r"gs://([^/]+)/(.+)", gcs_destination_uri)
    prefix = match.group(2)
    bucket = storage_client.get_bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))

    def extract_page_number(blob):
        match = re.search(r'output-(\d+)-to-\d+', blob.name)
        if match:
            return int(match.group(1))
        return -1

    sorted_blobs = sorted(blobs, key=extract_page_number)

    complete_text = ""
    for output in sorted_blobs:
        if not output.name.endswith("/"):  # Filter out directories
            json_string = output.download_as_bytes().decode("utf-8")
            response = json.loads(json_string)

            for page_response in response.get('responses', []):
                if 'fullTextAnnotation' in page_response:
                    annotation = page_response['fullTextAnnotation']
                    complete_text += annotation.get('text', '')

    return complete_text