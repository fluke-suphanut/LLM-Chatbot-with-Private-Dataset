from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
import os

with open("agnos_forum_posts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [f"{post['title']} {post['content']}" for post in data if post.get("content")]

print("กำลังโหลดโมเดล")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("กำลังสร้าง")
embeddings = model.encode(texts, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype="float32"))

os.makedirs("vector_index", exist_ok=True)

faiss.write_index(index, "vector_index/faiss_index.idx")
with open("vector_index/metadata.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("เสร็จสิ้น")
