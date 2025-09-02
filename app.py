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

# Not run
class DefChromaEF(Embeddings):
  def __init__(self,ef):
    self.ef = ef

  def embed_documents(self,texts):
    return self.ef(texts)

  def embed_query(self, query):
    return self.ef([query])[0]

def extract_metadata(collection):
       
    # Get the first n embeddings from the collection
    col = client.get_or_create_collection(collection)
    
    # Extract the text from the embeddings
    # page = col.query(query_texts=["Abstract"])
    page = col.get(ids=["0"])
    return page["documents"][0][:577]
    
# Get list of collections from ChromaDB cloud
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
      "Summarize the text. Identify main theme of the text and the key findings and list theme and finding as a JSON format",
      "Give the abstract from the article",
      "What data analysis mentioned in the text",
      "What has been studied on this topic?",
      "What are the key findings and conclusions from the text", 
      "Suggest a future research direction based on the text",
      "What are the gaps in knowledge or inconsistencies in the text?",
      "What are the problem statements mentioned in the text?",
      "What are the limitations mentioned in the text?",
      "What are the factors that influence students' attitudes towards ChatGPT?",
      "Give theoretical research framework from the text",
      "Highlight important previous studies that related to the text"]

st.title("Knowledge Base")

# Create a selectbox to choose a collection
mt = ""
selected_collection = st.sidebar.selectbox("Select a collection", pdf)
selected_qs = st.sidebar.selectbox("Select a question", qs)
st.write(selected_collection)

db = Chroma(client=client, collection_name=selected_collection)
retriever = db.as_retriever()

template = """<bos><start_of_turn>user\nAnswer the question based only on the following context and extract out a meaningful answer. \
Please write in full sentences with correct spelling and punctuation. if it makes sense use lists. \
Use the JSON format as given if required \
{{
  "theme": "Effect of ChatGPT-based project-based learning model on news text writing skills",
  "main_findings": [
    {{
      "heading": "Effect of ChatGPT-based project-based learning model",
      "description": "The study found that using the ChatGPT-based project-based learning model greatly enhances the capacity to produce news articles."
    }},
    {{
      "heading": "Interaction between ChatGPT-based project-based learning model and digital literacy",
      "description": "There is no interaction between the ChatGPT-based project-based learning model and digital literacy in influencing the capacity to create news material."
    }},
    {
      "heading": "Digital literacy's impact on news text writing skills",
      "description": "Digital literacy does not have a significant impact on news text writing skills when used in conjunction with the ChatGPT-based project-based learning model."
    },
    {{
      "heading": "Significance of comprehensive strategy for fostering news text writing abilities",
      "description": "The study emphasizes the significance of a comprehensive strategy for fostering students' news text writing abilities."
    }}
  ]
}}
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
    response_placeholder = st.empty()
    for r in rag_chain.stream(question):
        response += r.content
    return response

if __name__ == "__main__":
    user_question = st.text_input("Ask a question (or close tab to exit): ", value = selected_qs)
    if user_question.lower() == 'quit':
        exit()
    answer = st.write(ask_question(user_question))
