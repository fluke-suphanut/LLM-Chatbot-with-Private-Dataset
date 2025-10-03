import os, json, faiss, numpy as np, requests, streamlit as st
from sentence_transformers import SentenceTransformer

INDEX_PATH = "vector_index/faiss_index.idx"
META_PATH  = "vector_index/metadata.json"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL  = os.environ.get("OLLAMA_MODEL", "llama3:8b")

MAX_CONTEXT_CHARS = 700
TOP_K_DEFAULT     = 2

SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยทางการแพทย์ภาษาไทย ตอบสั้น กระชับ ชัดเจน "
    "ตอบเป็นภาษาไทยเท่านั้น ห้ามใช้ภาษาอื่น อ้างอิงเฉพาะข้อมูลใน CONTEXT "
    "หากข้อมูลไม่พอให้บอกตรง ๆ และแนะนำให้พบแพทย์ตามความเหมาะสม "
    "หลีกเลี่ยงการวินิจฉัยเฉพาะเจาะจงและห้ามระบุชื่อยาเฉพาะเว้นแต่พบในบริบท"
)

st.set_page_config(page_title="Agnos RAG Chat", layout="centered")
st.title("Agnos Health Forum")
with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("จำนวนเอกสารอ้างอิง", 1, 5, TOP_K_DEFAULT)
    max_chars = st.slider("ตัดความยาวต่อเอกสาร", 300, 2000, MAX_CONTEXT_CHARS, step=100)
    model_name = st.text_input("Ollama model", value=LLM_MODEL, help="llama3:8b, mistral:7b, tinyllama")
    if st.button("ล้างประวัติการคุย"):
        st.session_state.clear()
        st.rerun()

if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
    st.error("ไม่พบไฟล์")
    st.stop()

try:
    ok = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
    assert ok.status_code == 200
except Exception as e:
    st.error(f"ต่อ Ollama ไม่ได้ที่ {OLLAMA_URL} — ติดตั้ง/เปิด Ollama{model_name}`\n\n{e}")
    st.stop()

index = faiss.read_index(INDEX_PATH)
metadata = json.load(open(META_PATH, "r", encoding="utf-8"))

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")
embedder = load_embedder()

def ollama_generate(prompt: str, model: str, max_new_tokens=320, temperature=0.6, top_p=0.95):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_new_tokens, "temperature": temperature, "top_p": top_p}
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def retrieve(query: str, k: int, max_chars: int):
    qv = embedder.encode([query])
    D, I = index.search(np.array(qv, dtype="float32"), k)
    chunks, refs = [], []
    for idx in I[0]:
        if 0 <= idx < len(metadata):
            title = metadata[idx].get("title", "")
            url   = metadata[idx].get("url", "")
            content = (metadata[idx].get("content") or "")[:max_chars]
            refs.append({"title": title, "url": url})
            chunks.append(f"ชื่อเรื่อง: {title}\nที่มา: {url}\nเนื้อหา: {content}")
    return "\n\n".join(chunks), refs

def build_answer_prompt(history: list[dict], context_text: str, question: str) -> str:
    hist_txt = ""
    for turn in history[-6:]:
        role = "ผู้ใช้" if turn["role"] == "user" else "ผู้ช่วย"
        hist_txt += f"{role}: {turn['content']}\n"
    return (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
        f"[ประวัติการสนทนาล่าสุด]\n{hist_txt or '—'}\n\n"
        f"[CONTEXT]\n{context_text}\n\n"
        f"[คำถามผู้ใช้]\n{question}\n\n"
        "โปรดตอบเป็นภาษาไทยเท่านั้น โดยสรุปสั้นๆ"
        "และเตือนพบแพทย์เมื่อมีอาการอันตราย\n\n"
        "[คำตอบ]\n"
    )

def build_rewrite_prompt(history: list[dict], question: str) -> str:
    hist_txt = ""
    for turn in history[-6:]:
        role = "ผู้ใช้" if turn["role"] == "user" else "ผู้ช่วย"
        hist_txt += f"{role}: {turn['content']}\n"
    return (
        "จงเขียนคำถามใหม่ให้เป็นประโยคเดี่ยวที่สมบูรณ์และชัดเจนสำหรับการค้นหา โดยยึดบริบทจากประวัติการคุยด้านล่าง "
        "ห้ามตอบอย่างอื่น ให้คืนค่าเป็นคำถามเพียงบรรทัดเดียว (ภาษาไทย)\n\n"
        f"[ประวัติ]\n{hist_txt}\n"
        f"[คำถามล่าสุด]\n{question}\n\n"
        "คำถามที่สรุปแล้ว:"
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "สวัสดีค่ะ/ครับ ถามอาการได้เลยครับ"}
    ]

for m in st.session_state["messages"]:
    with st.chat_message("assistant" if m["role"]=="assistant" else "user"):
        st.markdown(m["content"])

user_msg = st.chat_input("พิมพ์คำถาม")
if user_msg:
    st.session_state["messages"].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.spinner("กำลังสรุปคำถาม"):
        try:
            rewritten = ollama_generate(build_rewrite_prompt(st.session_state["messages"], user_msg),
                                        model=model_name, max_new_tokens=64, temperature=0.2)
        except Exception as e:
            rewritten = user_msg
    with st.spinner("กำลังค้นหาบริบท"):
        context_text, refs = retrieve(rewritten, k=top_k, max_chars=max_chars)
    with st.spinner("กำลังสรุปคำตอบ"):
        answer = ollama_generate(build_answer_prompt(st.session_state["messages"], context_text, rewritten),
                                 model=model_name, max_new_tokens=320, temperature=0.6)

    with st.chat_message("assistant"):
        st.markdown(answer)
        if refs:
            with st.expander("แหล่งอ้างอิง"):
                for i, r in enumerate(refs, 1):
                    st.markdown(f"**{i}. [{r['title']}]({r['url']})**")

    st.session_state["messages"].append({"role": "assistant", "content": answer})
