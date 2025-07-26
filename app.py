import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

# Allow KMP errors to be ignored (needed for some systems)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Function to set a background image in Streamlit
def set_background(image_file):
    import base64
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        
        f"""
            <head>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari&display=swap" rel="stylesheet">
    </head>
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set your background image here — make sure this path is correct!
set_background(r"C:\Users\vagee\OneDrive\Attachments\Desktop\agentic ai\krish\langchain_with_openai\bhagwatgita\Copilot_20250726_004311.png")

# Load and prepare your LangChain QA model
@st.cache_resource
def load_qa_chain():
    loader = TextLoader("gita2.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    split_docs = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.from_documents(split_docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = OllamaLLM(model="mistral")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    return qa

qa_chain = load_qa_chain()

# Inject CSS for blur container, text colors, input & button styling
st.markdown(
    """
    <style>
    
    /* Background image set above */

  

    /* Title styling */
    .project-title {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color:white;  /* Gold color */
        margin-bottom: 10px;
    }

    /* Subtitle text */
    .project-subtitle {
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 30px;
        color: white;
    }

    /* Style for input box */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.85) !important;
        color: black  !important;
        border-radius: 8px !important;
        font-size: 1.5rem !important;
        padding: 10px !important;
    }

    /* Style for the primary button */
    button[kind="primary"] {
        background-color: #ffd700 !important;
        color: black !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.5rem;
    }

    /* Chat box style */
    .chat {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-top: 20px;
        white-space: pre-wrap;
        word-break: break-word; /* Add this */
    font-family: "Noto Sans Devanagari", "Mukta", sans-serif; /* Add Devanagari fallback */
    }
    input[type="text"] {
        color: black !important;
        background-color: white !important;
        border: 1px solid #ccc !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Start the transparent blurred container
st.markdown('<div class="blur-container">', unsafe_allow_html=True)

# Title and subtitle
st.markdown('<h1 class="project-title">Bhagavad Gita AI — Wisdom from Kurukshetra</h1>', unsafe_allow_html=True)
st.markdown('<p class="project-subtitle">Ask questions about Dharma, Karma, Life & Duty — directly from the Gita</p>', unsafe_allow_html=True)

# User input
query = st.text_input("Ask your question to Lord Krishna", placeholder="e.g. What is true duty according to the Gita?")

# Button for submitting (optional: can just use Enter in input)
if st.button("Submit"):
    if query:
        with st.spinner("🧠 Thinking like Krishna..."):
            result = qa_chain.invoke(query)
            st.markdown("<div class='chat'>", unsafe_allow_html=True)
            st.markdown(f"**Your Question:** {query}", unsafe_allow_html=True)
            st.markdown(f"**Krishna's Wisdom:** {result['result']}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a question before submitting.")

# End blurred container
st.markdown('</div>', unsafe_allow_html=True)
