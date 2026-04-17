import pandas as pd
import pymongo
from dotenv import load_dotenv
import os

load_dotenv()

df = pd.read_csv(r"G:\My Drive\AI_ML LEARNING\MLOPs\PROJECTS\vehicle_insurance_mlops_project\notebook\data.csv")
print(df.head())

# df should be converted into dict before we push it to mongodb

data = df.to_dict(orient='records')
# data

DB_NAME = "Proj1"
COLLECTION_NAME = "Proj1-Data"
CONNECTION_URL = os.getenv("CONNECTION_URL_ID")
# please either remove your credentials or delete the mongoDB resource bofore pushing it to github.

client = pymongo.MongoClient(CONNECTION_URL)
data_base = client[DB_NAME]
collection = data_base[COLLECTION_NAME]

# Uploading data to MongoDB
rec = collection.insert_many(data)

print(rec.inserted_ids)