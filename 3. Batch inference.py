# Databricks notebook source
# MAGIC %md
# MAGIC ### Set Model Serving endpoint

# COMMAND ----------

model_endpoint = 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/tiffcv/invocations'

# COMMAND ----------

import mlflow
import mlflow.pytorch
import torch
import torchvision.transforms as transforms
from PIL import Image
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
import io


mlflow.set_registry_uri('databricks-uc')

# Load the trained model from MLflow using a version number or alias
model_uri = "models:/mpelletier.geospatial.tiffcv@champion"  # Correctly specify the alias
model = mlflow.pytorch.load_model(model_uri)
model.eval()  # Set the model to evaluation mode

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to preprocess images
def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = transform(image)
    return image

# Define a Pandas UDF for batch inference
@pandas_udf("int", PandasUDFType.SCALAR)
def batch_inference_udf(image_data_series: pd.Series) -> pd.Series:
    images = torch.stack([preprocess_image(image_data) for image_data in image_data_series])
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    return pd.Series(preds.cpu().numpy())

# Load the table containing image data
image_table = spark.read.table("mpelletier.geospatial.tiff")

# Perform batch inference
result_df = image_table.withColumn("prediction", batch_inference_udf(image_table["content"]))

# Display the results
display(result_df)
