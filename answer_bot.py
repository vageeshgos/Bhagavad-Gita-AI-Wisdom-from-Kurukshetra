import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from  langchain_community.vectorstores import FAISS
from  langchain_ollama import OllamaEmbeddings
from  langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
print("Loading Bhagavad Gita text...")
loader=TextLoader("gita2.txt",encoding="utf-8")
doc=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=30)
docs=text_splitter.split_documents(doc)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db=FAISS.from_documents(docs,embeddings)
#RetrievalQA helps the bot find the right shloka or explanation from Gita or Sanskrit texts when a user asks a question.
#RetrievalQA uses the llama embedding model to create a vector representation of the documents and then uses that vector representation to find the most relevant documents to the user's 
#Ollama runs the local language model (like LLaMA3) to understand the question and generate a clear, human-like answer in your chosen language."""
#Create the retriever
retriever = db.as_retriever(search_kwargs={"k": 3})  # Only fetch top 3 relevant chunks


llm = OllamaLLM(model="mistral")
#Create the question answering function
qa=RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")#stuffed" (combined) into a single text block and given to the LLM at once.
"""🧠 Example:
Imagine the bot finds 3 related paragraphs from the Gita.
With stuff, it does this:

diff
Copy code
Input to LLM:
[Q] What does Gita say about duty?
[Docs]
- Paragraph 1...
- Paragraph 2...
- Paragraph 3...
Then LLM reads all 3 at once and answers.
-------> But if your text is too long (over token limit), use other types like map_reduce or refin"""
print("\n Welcome to the Bhagavad Gita AI Bot ")
print("Ask any question about life, karma, duty, or Gita's teachings.")
print("Type 'exit' to quit.\n")

while True:
    query=input("ask a question from  gita(or type'exit'):")
    if query.lower()=="exit":
        print("thank you for using our bot")
        break
    result=qa.invoke(query)
    print("\n answer:",result["result"])
    print("\n"+"--"*50)
    
