import os
import logging
import requests
import json
import pandas as pd


# Configure logging
log_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(module)s:%(funcName)s():%(lineno)d] %(message)s'
logging.basicConfig(
    format=log_format,
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

# Load the environment variables
from dotenv import load_dotenv
load_dotenv()
hugging_face_api_token = os.environ["HUGGING_FACE_API_TOKEN"]

# Determine the current directory and the extract path
current_dir = os.path.abspath(os.path.dirname(__file__))
extract_file = "extract.txt"
extract_path = os.path.join(current_dir, extract_file)

# Read the text from the file
with open(extract_path, "r") as fp:
    text = fp.read()

# Divide the document text into chunks of equal word count
words = text.split(" ")
word_count = len(words)
chunk_count = 10
chunks = []
chunk_size = round(word_count / chunk_count)
for i in range(0, chunk_count - 1):
    chunk = " ".join(words[i * chunk_size:(i+1)*chunk_size])
    chunks.append(chunk)
chunks.append(" ".join(words[chunk_count - 1 * chunk_size:]))

# Write the chunks to a file
chunk_file = "chunks.json"
chunk_file_path = os.path.join(current_dir, chunk_file)
with open(chunk_file_path, "w") as fp:
    json.dump(chunks, fp, indent=4)

# Convert the text into embeddings using hugging face api
model_id = "sentence-transformers/all-MiniLM-L6-v2"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hugging_face_api_token}"}

def query(texts):
    response = requests.post(api_url, 
                             headers=headers, 
                             json={"inputs": texts, "options":{"wait_for_model":True}},
                             verify=False)
    return response.json()

embeddings_list = query(chunks)

# Load embeddings into a dataframe
embeddings_df = pd.DataFrame(embeddings_list)

# Write the embeddings df to file
embeddings_file = "embeddings.csv"
embeddings_path = os.path.join(current_dir, embeddings_file)
embeddings_df.to_csv(embeddings_path)
