import os
import json
import faiss
import numpy as np
import hashlib
import importlib
from tools import generated_tools
from sentence_transformers import SentenceTransformer

# Config
SCHEMA_PATH = "tools/functions_schema.json"
VECTOR_DIR = "vector"
os.makedirs(VECTOR_DIR, exist_ok=True)

INDEX_PATH = os.path.join(VECTOR_DIR, "tools.index")
EMBEDDINGS_PATH = os.path.join(VECTOR_DIR, "tool_embeddings.npy")
HASH_PATH = os.path.join(VECTOR_DIR, "tools.hash")
DIM = 768  # Embedding size

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute hash of schema
def compute_schema_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Load schema
with open(SCHEMA_PATH, "r") as f:
    functions_schema = json.load(f)

current_hash = compute_schema_hash(SCHEMA_PATH)
stored_hash = ""

if os.path.exists(HASH_PATH):
    with open(HASH_PATH, "r") as f:
        stored_hash = f.read().strip()

# Decide whether to rebuild index
need_rebuild = (current_hash != stored_hash) or not os.path.exists(INDEX_PATH)

if need_rebuild:
    print("🛠️ Rebuilding FAISS index and embeddings...")
    importlib.reload(generated_tools)
    # Compute embeddings from tool descriptions
    embeddings = model.encode(
        [tool["description"] for tool in functions_schema],
        show_progress_bar=True
    )
    embeddings = np.array(embeddings).astype("float32")  # type: ignore # Ensure float32 type
    
    # Save to disk
    np.save(EMBEDDINGS_PATH, embeddings)
    
    # Dynamically set dimension based on actual embeddings
    DIM = embeddings.shape[1]
    index = faiss.IndexFlatL2(DIM)
    
    # Add embeddings to index
    index.add(embeddings) # type: ignore
    faiss.write_index(index, INDEX_PATH)
    
    # Save hash
    with open(HASH_PATH, "w") as f:
        f.write(current_hash)
else:
    print("✅ Using cached FAISS index and embeddings.")
    embeddings = np.load(EMBEDDINGS_PATH)
    DIM = embeddings.shape[1]  # Ensure DIM is defined in this path too
    index = faiss.read_index(INDEX_PATH)


# Search function
def search_tools(query: str, top_k=5):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding).astype("float32"), top_k) # type: ignore
    return [functions_schema[i] for i in indices[0]]
