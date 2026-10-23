# from chromadb.utils import embedding_functions
# from langchain_core.output_parsers import StrOutputParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import streamlit as st
import chromadb
import os

st.title("Knowledge Base")

load_dotenv()
os.environ["OPENROUTER_API_KEY"] = os.environ.get('OPENROUTE_API_KEY')

@st.cache_resource
def get_llm():
    return init_chat_model("nvidia/nemotron-3-ultra-550b-a55b:free", model_provider="openrouter")

llm = get_llm()

ef = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()

class DefChromaEF(Embeddings):
  def __init__(self,ef):
    self.ef = ef

  def embed_documents(self,texts):
    return self.ef(texts)

  def embed_query(self, query):
    return self.ef([query])[0]

def extract_metadata(collection):
    col = client.get_or_create_collection(collection)
    page = col.get(where={"page_number": 1}, limit=1, include=["metadatas"])
    if page["metadatas"]:
        return page["metadatas"][0] or {}
    return {}

def load_collection_metadata(collection_names, progress):
    metadata = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(extract_metadata, name): name for name in collection_names}
        for index, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                metadata[name] = future.result()
            except Exception:
                metadata[name] = {}
            progress.progress(index / len(collection_names))
    return metadata
    
# Get list of collections from ChromaDB
# client = chromadb.PersistentClient(path="../chromadb")
# Accessing .env
ch_api_key = os.getenv('CHROMA_API_KEY')
ch_tenant = os.getenv('CHROMA_TENANT')
ch_database = os.getenv('CHROMA_DATABASE')

@st.cache_resource
def get_chroma_client(tenant, database, api_key):
    return chromadb.CloudClient(
        tenant=tenant,
        database=database,
        api_key=api_key
    )

client = get_chroma_client(ch_tenant, ch_database, ch_api_key)

@st.cache_data(ttl=300)
def list_collection_names():
    return [collection.name for collection in client.list_collections()]

with st.spinner("Connecting to collection store..."):
    pdf = list_collection_names()
if not pdf:
    st.info("No collections are available.")
    st.stop()

collection_names = tuple(pdf)
if st.session_state.get("collection_names") != collection_names:
    with st.status(f"Loading details for {len(pdf)} collections...", expanded=False) as status:
        progress = st.progress(0)
        metadata_by_collection = load_collection_metadata(collection_names, progress)
        st.session_state["collection_names"] = collection_names
        st.session_state["metadata_by_collection"] = metadata_by_collection
        status.update(label=f"Loaded details for {len(pdf)} collections", state="complete")
else:
    metadata_by_collection = st.session_state["metadata_by_collection"]
qs = ["Summarize the text",
      "Give the abstract from the article",
      "What data analysis mentioned in the text",
      "What has been studied on this topic?",
      "describe the psychological, attitude, cognitive and behavior according to text",
      "What are the key findings and conclusions from the text", 
      "Suggest a future research direction based on the text",
      "What are the gaps in knowledge or inconsistencies in the text?",
      "What are the problem statements mentioned in the text?",
      "What are the limitations mentioned in the text?",
      "Give theoretical research framework from the text",
      "Highlight important previous studies that related to the text"]

# Create a selectbox to choose a collection
mt = ""
selected_collection = st.sidebar.selectbox(
    "Select a collection",
    pdf,
    format_func=lambda name: metadata_by_collection[name].get("Title") or name,
)
selected_qs = st.sidebar.selectbox("Select a question", qs)
selected_metadata = metadata_by_collection[selected_collection]
document_title = selected_metadata.get("Title") or selected_collection
author = selected_metadata.get("Author")
year = selected_metadata.get("Year")
st.subheader(document_title)
if author:
    citation = f"({author}, {year})" if year else f"({author})"
    st.caption(citation)

db = Chroma(client=client, collection_name=selected_collection)
retriever = db.as_retriever()

template = """<bos><start_of_turn>user\nAnswer the question based only on the following context and extract out a meaningful answer. \
Please write in full sentences with correct spelling and punctuation. if it makes sense use lists. \
If the context doesn't contain the answer, just respond that you are unable to find an answer. \

CONTEXT: {context}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model\n
ANSWER:"""

prompt = ChatPromptTemplate.from_template(template)

answer_chain = prompt | llm


if __name__ == "__main__":
    with st.form("question_form"):
        user_question = st.text_input("Ask a question", value=selected_qs)
        submitted = st.form_submit_button("Search")
    if submitted and user_question.strip():
        with st.status("Searching the selected collection...", expanded=False) as search_status:
            documents = retriever.invoke(user_question)
            search_status.update(label=f"Retrieved {len(documents)} passages", state="complete")
        with st.status("Generating answer...", expanded=True) as answer_status:
            try:
                context = "\n\n".join(document.page_content for document in documents)
                st.write_stream(answer_chain.stream({"context": context, "question": user_question}))
                answer_status.update(label="Answer complete", state="complete")
            except Exception as error:
                answer_status.update(label="Answer generation failed", state="error")
                st.error(f"Could not complete the answer: {error}")
