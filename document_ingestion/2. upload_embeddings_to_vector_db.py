import os
import logging
import pandas
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import hashlib

# Configure logging
log_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(module)s:%(funcName)s():%(lineno)d] %(message)s'
logging.basicConfig(
    format=log_format,
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

# Load the environment variables
from dotenv import load_dotenv
load_dotenv()
qdrant_api_url = os.environ["QDRANT_API_URL"]
pdf_file_name = os.environ["PDF_FILE_NAME"]
qdrant_collection_name = os.environ["QDRANT_COLLECTION_NAME"]


# Load the embeddings
embeddings_file = "embeddings.csv"
current_dir = os.path.abspath(os.path.dirname(__file__))
embeddings_file_path = os.path.join(current_dir, embeddings_file)
embeddings = pandas.read_csv(embeddings_file_path, index_col=0)

# Conect to vectordb
client = QdrantClient(url=qdrant_api_url)

# Create a collection in the vectordb
# Specify the length of the vectors and the distance algorithm
if not client.collection_exists(qdrant_collection_name):
    client.create_collection(
        collection_name=qdrant_collection_name,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.DOT)
    )

# Create a unique ID for the file
pdf_id = hashlib.sha256(pdf_file_name.encode()).hexdigest()

# Upsert the embeddings
points = []
for i in range(0, embeddings.shape[0]):
    vector = embeddings.iloc[i]
    point = PointStruct(id=i, vector=vector, payload={"chunk_id": i, "pdf_id": pdf_id})
    points.append(point)
    
operation_info = client.upsert(
    collection_name=qdrant_collection_name,
    wait=True,
    points=points,
)

print(operation_info)
