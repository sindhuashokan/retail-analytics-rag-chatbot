import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Retail Analytics RAG Chatbot",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Retail Analytics RAG Chatbot")
st.write("Ask business questions about sales, customers, churn risk, products, and revenue insights.")

# -----------------------------
# Load knowledge base
# -----------------------------
@st.cache_data
def load_knowledge_base():
    with open("knowledge_base.txt", "r", encoding="utf-8") as file:
        text = file.read()

    chunks = text.split("\n\n")
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks

chunks = load_knowledge_base()

# -----------------------------
# Load embedding model
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# Create embeddings
# -----------------------------
@st.cache_resource
def create_embeddings(_chunks):
    embeddings = model.encode(_chunks)
    return np.array(embeddings)

embeddings = create_embeddings(chunks)

# -----------------------------
# Chat history
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# User input
# -----------------------------
user_question = st.chat_input("Ask a retail analytics question...")

if user_question:
    question_embedding = model.encode([user_question])
    similarity_scores = cosine_similarity(question_embedding, embeddings)[0]

    best_index = np.argmax(similarity_scores)
    best_score = similarity_scores[best_index]

    retrieved_context = chunks[best_index]

    if best_score < 0.25:
        bot_answer = (
            "I could not find a strong matching insight in the knowledge base. "
            "Please ask about sales, products, customers, churn risk, revenue, or monthly trends."
        )
    else:
        bot_answer = f"Based on the retail analytics knowledge base:\n\n{retrieved_context}"

    st.session_state.chat_history.append({
        "user": user_question,
        "bot": bot_answer,
        "score": best_score
    })

# -----------------------------
# Display chat
# -----------------------------
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["user"])

    with st.chat_message("assistant"):
        st.write(chat["bot"])
        st.caption(f"Similarity Score: {chat['score']:.2f}")

# -----------------------------
# Clear chat
# -----------------------------
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()