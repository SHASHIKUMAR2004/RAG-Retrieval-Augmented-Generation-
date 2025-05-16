from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import uuid

# Initialize the Qdrant client and the SentenceTransformer
qdrant_client = QdrantClient(url="YOUR_QDRANT_URL", api_key="YOUR_QDRANT_APIKEY")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the CSV file (ensure the CSV is in the same directory as your script or provide full path)
df = pd.read_csv("chunks.csv")

# Ensure Qdrant collection exists
collection_name = "rag_chunks"
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vector_size=384,
        distance="Cosine"
    )

# Function to upsert data into Qdrant
def upsert_to_qdrant(documents):
    for idx, row in tqdm(documents.iterrows(), total=documents.shape[0], desc="Upserting to Qdrant"):
        # Create embedding for each document
        embedding = model.encode([row['content']])[0]
        
        # Generate a unique string ID for the point
        point_id = str(uuid.uuid4())  # Convert UUID to string
        
        # Create a PointStruct object
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={"content": row["content"]}
        )
        
        # Upsert the point into Qdrant
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[point]
        )

# Call the upsert function
upsert_to_qdrant(df)
