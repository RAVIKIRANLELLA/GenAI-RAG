import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pathlib import Path

region = "us-east-1"  # or your Bedrock region

bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)

titan_embeddings = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id="amazon.titan-embed-text-v2:0"  # or v1
)

sonnet = BedrockChat(
    client=bedrock_runtime,
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
)

pdf_path = "../data/pdf/attention.pdf"

# 1) Load PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()


# 2) Chunk text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
docs = text_splitter.split_documents(documents)


# 3) Create FAISS vector store with Titan embeddings
vector_store = FAISS.from_documents(docs, titan_embeddings)


# 4) Persist index locally (optional)
vector_store.save_local("faiss_index")

# FAISS provides similarity search over Titan-generated embeddings, returning the most relevant chunks for a query.
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings=titan_embeddings,
    allow_dangerous_deserialization=True,  # required by FAISS loader
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

def build_prompt(question: str, context_docs):
    context_text = "\n\n".join(d.page_content for d in context_docs)
    return f"""You are a helpful assistant.

Use the following context to answer the question. If the answer is not in the context, say you don't know.

Context:
{context_text}

Question: {question}
Answer:"""

def ask_rag(question: str) -> str:
    docs = retriever.invoke(question)
    prompt = build_prompt(question, docs)

    resp = sonnet.invoke(prompt)
    return resp.content  # LangChain ChatModel returns .content

user_question = "can you briefly describe about the context of attention.pdf?"
answer = ask_rag(user_question)
print(answer)