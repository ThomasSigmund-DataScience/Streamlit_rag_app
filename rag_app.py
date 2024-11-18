
import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# Set environment variables (replace with your actual tokens)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
os.environ["GROQ_CLOUD"] = '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'

# Initialize LLM
llm = ChatGroq(temperature=0,
               groq_api_key=os.getenv('GROQ_CLOUD'),
               model_name="mixtral-8x7b-32768")

# Initialize embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "./"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                   cache_folder=embeddings_folder)

# Define the absolute paths to the FAISS index files
index_path = "C:\\faiss_index"
faiss_file = os.path.join(index_path, "index.faiss")
pkl_file = os.path.join(index_path, "index.pkl")

# Check if the FAISS index files exist
if not os.path.isfile(faiss_file) or not os.path.isfile(pkl_file):
    raise FileNotFoundError(f"FAISS index files not found in {index_path}")

# Load Vector Database
vector_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Initialize retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# Initialize memory
@st.cache_resource
def init_memory(_llm):
    return ConversationBufferMemory(
        llm=llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)

memory = init_memory(llm)

# Define prompt template
template = """You are a nice chatbot having a conversation with a human. Answer the question based only on the following context and previous conversation. Keep your answers short and succinct.

Previous conversation:
{chat_history}

Context to answer question:
{context}

New human question: {question}
Response:"""

prompt = PromptTemplate(template=template,
                        input_variables=["context", "question"])

# Initialize chain
chain = ConversationalRetrievalChain.from_llm(llm,
                                              retriever=retriever,
                                              memory=memory,
                                              return_source_documents=True,
                                              combine_docs_chain_kwargs={"prompt": prompt})

# Streamlit app
st.title("CHAT: An Introduction to Statistical Learning___PYTHON")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Curious minds wanted!"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Going down the decision tree for answers ..."):
        # Send question to chain to get answer
        answer = chain.invoke(prompt)

        # Extract answer from dictionary returned by chain
        response = answer["answer"]

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer["answer"])

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
