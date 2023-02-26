


BUCKET_NAME = "gs://<YOUR-BUCKET-NAME>"  # @param {type:"string"}
REGION = "us-central1"  # @param {type:"string"}

'''
! gsutil mb -l $REGION $BUCKET_NAME
! gsutil uniformbucketlevelaccess set on gs://<YOUR-BUCKET-NAME>
'''


# Authenticate your google Cloud account
import os
import sys

# If you are running this notebook in Colab, run this cell and follow the
# instructions to authenticate your GCP account. This provides access to your
# Cloud Storage bucket and lets you submit training jobs and prediction
# requests.

# The Google Cloud Notebook product has specific requirements
IS_GOOGLE_CLOUD_NOTEBOOK = os.path.exists("/opt/deeplearning/metadata/env_version")

# If on Google Cloud Notebooks, then don't execute this code
if not IS_GOOGLE_CLOUD_NOTEBOOK:
    if "google.colab" in sys.modules:
        from google.colab import auth as google_auth

        google_auth.authenticate_user()

    # If you are running this notebook locally, replace the string below with the
    # path to your service account key and run this cell to authenticate your GCP
    # account.
    elif not os.getenv("IS_TESTING"):
        %env GOOGLE_APPLICATION_CREDENTIALS 



### create BigQuery dataset ###

from datetime import datetime
from google.cloud import bigquery

# Output dataset
DESTINATION_DATA_SET = "movie_predictions"  # @param {type:"string"}
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
DESTINATION_DATA_SET = "{prefix}_{timestamp}".format(
    prefix=DESTINATION_DATA_SET, timestamp=TIMESTAMP
)

# Output table. Make sure that the table does NOT already exist; the BatchReadFeatureValues API cannot overwrite an existing table
DESTINATION_TABLE_NAME = "training_data"  # @param {type:"string"}

DESTINATION_PATTERN = "bq://{project}.{dataset}.{table}"
DESTINATION_TABLE_URI = DESTINATION_PATTERN.format(
    project=PROJECT_ID, dataset=DESTINATION_DATA_SET, table=DESTINATION_TABLE_NAME
)

# Create dataset
REGION = "us-central1"  # @param {type:"string"}
client = bigquery.Client()
dataset_id = "{}.{}".format(client.project, DESTINATION_DATA_SET)
dataset = bigquery.Dataset(dataset_id)
dataset.location = REGION
dataset = client.create_dataset(dataset, timeout=30)
print("Created dataset {}.{}".format(client.project, dataset.dataset_id))



### feature creation example ###

admin_client.batch_create_features(
    parent=admin_client.entity_type_path(PROJECT_ID, REGION, FEATURESTORE_ID, "users"),
    requests=[
        featurestore_service_pb2.CreateFeatureRequest(
            feature=feature_pb2.Feature(
                value_type=feature_pb2.Feature.ValueType.INT64,
                description="User age",
            ),
            feature_id="age",
        ),
        featurestore_service_pb2.CreateFeatureRequest(
            feature=feature_pb2.Feature(
                value_type=feature_pb2.Feature.ValueType.STRING,
                description="User gender",
                monitoring_config=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig(
                    snapshot_analysis=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig.SnapshotAnalysis(
                        disabled=True,
                    ),
                ),
            ),
            feature_id="gender",
        ),
        featurestore_service_pb2.CreateFeatureRequest(
            feature=feature_pb2.Feature(
                value_type=feature_pb2.Feature.ValueType.STRING_ARRAY,
                description="An array of genres that this user liked",
            ),
            feature_id="liked_genres",
        ),
    ],
).result()



### importing features values example ###


import_users_request = featurestore_service_pb2.ImportFeatureValuesRequest(
    entity_type=admin_client.entity_type_path(
        PROJECT_ID, REGION, FEATURESTORE_ID, "users"
    ),
# Make sure to replace your bucket name here.
    avro_source=io_pb2.AvroSource(
        # Source
        gcs_source=io_pb2.GcsSource(
            uris=[
                "gs://<Your-bucket-name>/users.avro"
            ]
        )
    ),
    entity_id_field="user_id",
    feature_specs=[
        # Features
        featurestore_service_pb2.ImportFeatureValuesRequest.FeatureSpec(id="age"),
        featurestore_service_pb2.ImportFeatureValuesRequest.FeatureSpec(id="gender"),
        featurestore_service_pb2.ImportFeatureValuesRequest.FeatureSpec(
            id="liked_genres"
        ),
    ],
    feature_time_field="update_time",
    worker_count=10,
)
# Start to import, will take a couple of minutes
ingestion_lro = admin_client.import_feature_values(import_movie_request)