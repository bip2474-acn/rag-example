import os
import logging
import requests
import json
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
redis_api_host = os.environ["REDIS_API_HOST"]
redis_api_port = os.environ["REDIS_API_PORT"]

# Determine the current directory
current_dir = os.path.abspath(os.path.dirname(__file__))


# Load the chunks
chunk_file = "chunks.json"
chunk_file_path = os.path.join(current_dir, chunk_file)
with open(chunk_file_path, "r") as fp:
    chunks = json.load(fp)

# Create a unique ID for the file
pdf_id = hashlib.sha256(pdf_file_name.encode()).hexdigest()

# Upload the chunks to the kv store
import redis
from redis.commands.search.field import TextField
r = redis.Redis(host=redis_api_host, port=redis_api_port)

doc = {
    "name": pdf_file_name,
    "chunks": chunks
}

r.json().set(f"docs:{pdf_id}", "$", doc)

# Grab a specific chunk
# r.json().get(f"docs:{pdf_id}")["chunks"][3]
