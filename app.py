import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient

# --- Page Config ---
st.set_page_config(page_title="Nyaya.GPT", page_icon="⚖️", layout="wide")

# --- Load HF Token ---
if "HF_TOKEN" in st.secrets:
    hf_token = st.secrets["HF_TOKEN"]
else:
    st.error("HF_TOKEN not found in .streamlit/secrets.toml!")
    st.stop()

st.title("⚖️ Nyaya.GPT — Indian Legal AI Assistant")
st.markdown("---")

# --- Load RAG System ---
@st.cache_resource
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    if not os.path.exists("./local_chroma_db"):
        st.error("⚠️ Database not found! Please run 'python main.py' first.")
        st.stop()

    vector_db = Chroma(
        persist_directory="./local_chroma_db",
        embedding_function=embeddings
    )

    client = InferenceClient(
        provider="auto",
        api_key=hf_token,
    )

    return vector_db, client


def format_docs(docs):
    formatted = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get('source', 'Unknown'))
        content = f"Source: {source}\nContent: {doc.page_content}"
        formatted.append(content)
    return "\n\n---\n\n".join(formatted)


def get_sources(docs):
    return list({os.path.basename(d.metadata.get('source', 'Unknown')) for d in docs})


# --- System Prompt ---
SYSTEM_PROMPT = """You are a friendly yet professional Legal Guide. Your job is to take complex Indian laws and explain them simply for a common person.

STRUCTURE RULES:
1. START with the heading **EASY SUMMARY:** followed by a 1-2 sentence simple explanation.
2. FOLLOW with the heading **LEGAL DESCRIPTION:** which includes specific Sections and technical details.
3. USE bolding for Sections (e.g., **Section 34**).
4. If the context doesn't have the answer, say "I don't have that specific document in my library yet."

EXAMPLES:

Example 1:
Question: "If I just stood outside while my friend stole a phone, am I in trouble?"
Answer:
**EASY SUMMARY:** Yes, you are in trouble. Even if you didn't touch the phone, if you were part of the plan, the law treats your whole group as one team.
**LEGAL DESCRIPTION:** Under **Section 34 of the IPC**, when a criminal act is done by several persons in furtherance of a "Common Intention," every person is held equally responsible.

Example 2:
Question: "I bought a bike from a 15-year-old and he didn't give it to me. Can I sue him?"
Answer:
**EASY SUMMARY:** No, you cannot sue him. The law considers deals made with children (minors) to be invalid from the start.
**LEGAL DESCRIPTION:** Under **Section 11 of the Contract Act**, any agreement with a person under 18 is void and has no legal value.

Example 3:
Question: "Someone tried to punch me, so I hit them back with a stick. Is that okay?"
Answer:
**EASY SUMMARY:** Yes, you can defend yourself, but only use enough force to stop the attack.
**LEGAL DESCRIPTION:** **Section 97 of the IPC** gives you the right to protect your body. However, **Section 99** clarifies this right does not allow more harm than necessary for defense.

Example 4:
Question: "The doctor gave me the wrong injection because he was distracted. Is he liable?"
Answer:
**EASY SUMMARY:** Yes. If a professional is careless and it causes harm, they are legally responsible.
**LEGAL DESCRIPTION:** This falls under **Section 304A of the IPC** (Negligence), which punishes acts that are rash or ignore basic safety standards.

Example 5:
Question: "Someone tricked me into giving them money by making fake promises. Is this a crime?"
Answer:
**EASY SUMMARY:** Yes, this is a crime. Tricking someone into giving up money through lies is considered cheating.
**LEGAL DESCRIPTION:** **Section 420 of the IPC** deals with cheating and dishonestly inducing delivery of property.
"""

# --- Initialize ---
vector_db, hf_client = load_rag_system()
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 7, "fetch_k": 20}
)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input ---
if user_query := st.chat_input("Ask about Indian sections, articles, or legal advice..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Consulting legal database..."):
            try:
                # Retrieve relevant docs
                docs = retriever.invoke(user_query)
                context = format_docs(docs)
                sources = get_sources(docs)

                # Build messages
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{user_query}\n\nLEGAL ADVICE:"
                    }
                ]

                # Call Llama via HF Inference API
                response = hf_client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct",
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.1,
                )

                answer = response.choices[0].message.content
                st.markdown(answer)

                # Show sources
                if sources:
                    st.caption(f"📚 Sources: {', '.join(sources)}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Error: {e}")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("**Model:** `Llama-3.3-70B-Instruct`")
    st.write("**Provider:** HuggingFace Inference API")
    st.write("**Retriever:** MMR (k=7)")
    st.markdown("---")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()