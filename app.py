# from chromadb.utils import embedding_functions
# from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import streamlit as st
import chromadb
import os

load_dotenv()
os.environ["GROQ_API_KEY"] = os.environ.get('GROQ_API_KEY')
llm = init_chat_model("llama-3.1-8b-instant", model_provider="groq")

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
    # Get metadata fields we need
    metadata = {
        "Year": "",
        "Title": "",
        "Author": "",
        "Publication": "",
        "Publisher": ""
    }
    
    # Debugging - Show collection info
    # st.write(f"Collection name: {collection}")
    # st.write(f"Collection metadata: {col.metadata}")
    
    # Get first document that has metadata
    results = col.get(where={"page_number": 1})
    # st.write(f"Metadata query results: {results}")  # Debugging output
    
    if results["metadatas"]:
        # Extract metadata from first document with metadata
        doc_metadata = results["metadatas"][0]
        # st.write(f"Raw metadata document: {doc_metadata}")  # Debugging output
        
        try:
            # Directly assign from the metadata dictionary
            metadata["Title"] = doc_metadata.get("Title", "")
            metadata["Author"] = doc_metadata.get("Author", "")
            metadata["Publication"] = doc_metadata.get("Publication", "")
            metadata["Year"] = doc_metadata.get("Year", "")
            metadata["Publisher"] = doc_metadata.get("Publisher", "")
        except Exception as e:
            st.error(f"Error parsing metadata: {e}")
            print(f"Error parsing metadata: {e}")
    
    return metadata
    
# Get list of collections from ChromaDB
# client = chromadb.PersistentClient(path="../chromadb")
# Accessing .env
ch_api_key = os.getenv('CHROMA_API_KEY')
ch_tenant = os.getenv('CHROMA_TENANT')
ch_database = os.getenv('CHROMA_DATABASE')

# Retrieve Chroma cloud
client = chromadb.CloudClient(
    tenant=ch_tenant,
    database=ch_database,
    api_key=ch_api_key
)

collections = client.list_collections()
pdf = [collection.name for collection in collections]
qs = ["Summarize the text",
      "Give the abstract from the article",
      "What data analysis mentioned in the text",
      "What has been studied on this topic?",
      "What are the key findings and conclusions from the text", 
      "Suggest a future research direction based on the text",
      "What are the gaps in knowledge or inconsistencies in the text?",
      "What are the problem statements mentioned in the text?",
      "What are the limitations mentioned in the text?",
      "Give theoretical research framework from the text",
      "Highlight important previous studies that related to the text"]

st.title("Knowledge Base")

# Create a selectbox to choose a collection
mt = ""
selected_collection = st.sidebar.selectbox("Select a collection", pdf)
selected_qs = st.sidebar.selectbox("Select a question", qs)
# st.write(selected_collection)

# Extract and display metadata
metadata = extract_metadata(selected_collection)

st.subheader("Document Metadata")
st.text(f"Title: {metadata['Title']}")
st.text(f"Author(s): {metadata['Author']}")
st.text(f"Publication: {metadata['Publication']}")
st.text(f"Year: {metadata['Year']}")
st.text(f"Publisher: {metadata['Publisher']}")

db = Chroma(client=client, collection_name=selected_collection, embedding_function=DefChromaEF(ef))
retriever = db.as_retriever()

template = """<bos><start_of_turn>user\nAnswer the question based only on the following context and extract out a meaningful answer. \
Please write in full sentences with correct spelling and punctuation. if it makes sense use lists. \
When ask for JSON format, use the example given \
    {{ \
    "mindmap": {{ \
        "root": {{ \
        "id": "mainIdea", \
        "label": "Central Idea", \
        "children": [ \
            {{ \
            "id": "subIdea1", \
            "label": "Sub-Idea 1", \
            "children": [ \
                {{ \
                "id": "detail1a", \
                "label": "Detail 1a"
                }}, \
                {{ \
                "id": "detail1b", \
                "label": "Detail 1b" \
                }} \
            ] \
            }}, \
            {{ \
            "id": "subIdea2", \
            "label": "Sub-Idea 2", \
            "description": "More information about Sub-Idea 2" \
            }} \
        ] \
        }} \
    }} \
    }} \
If the context doesn't contain the answer, just respond that you are unable to find an answer. \

CONTEXT: {context}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model\n
ANSWER:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)


def ask_question(question):
    response = ""

    for r in rag_chain.stream(question):
        response += r.content
    return response

if __name__ == "__main__":
    user_question = st.text_input("Ask a question (or close tab to exit): ", value = selected_qs)
    if user_question.lower() == 'quit':
        exit()
    answer = st.write(ask_question(user_question))
