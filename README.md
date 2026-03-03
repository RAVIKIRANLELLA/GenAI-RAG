In the context of search systems (such as vector databases, search engines, or RAG frameworks), common search types include:

Keyword Search: Finds documents containing specific words or phrases.
Semantic Search: Uses embeddings to find semantically similar content.
Hybrid Search: Combines keyword and semantic search for better results.
Fuzzy Search: Matches terms that are similar but not identical (handles typos).
Boolean Search: Uses logical operators (AND, OR, NOT) to refine queries.
Faceted Search: Allows filtering by categories or attributes.
Range Search: Finds results within a numeric or date range.
Geospatial Search: Finds results based on location or proximity.

1. Imports and Setup
You import all necessary libraries:

boto3: AWS SDK for Python, to interact with Amazon Bedrock.
Pathlib: For file path handling.
LangChain modules: For document loading, splitting, embeddings, vector store, and LLM interaction.
2. Configuration
You set up:

REGION: AWS region for Bedrock.
PDF_PATH: Path to the PDF you want to use as knowledge base.
INDEX_DIR: Directory to store the FAISS vector index.
TOP_K: Number of top similar chunks to retrieve.
You also instantiate a Bedrock runtime client using boto3.

3. Embeddings
You create an embeddings object using Amazon Titan Text Embeddings v2 via Bedrock. This will convert text chunks into high-dimensional vectors for similarity search.

4. LLM Setup
You instantiate a conversational LLM (Amazon Nova Micro) via Bedrock, with specified temperature and token limits.

5. Helper Functions
build_rag_prompt:
Constructs a prompt for the LLM, including the retrieved context and the user’s question. It instructs the LLM to answer only using the provided context.
ask_rag:
Given a question and a retriever, it:
Retrieves relevant document chunks.
Builds the prompt.
Sends the prompt to the LLM.
Returns the LLM’s answer.
6. Load and Split PDF
Loads the PDF using PyPDFLoader.
Splits the document into overlapping chunks (1000 characters each, 200 overlap) for better retrieval granularity.
7. Build & Persist FAISS Vector Store
Converts all chunks into embeddings.
Builds a FAISS vector store from these embeddings.
Saves the vector store locally for reuse.
8. Load Vector Store and Ask a Question
Loads the FAISS vector store from disk.
Creates a retriever object for similarity search (top K chunks).
Asks a sample question:
"Can you briefly describe the context of attention.pdf?"
Uses the RAG pipeline to retrieve relevant chunks, build the prompt, query the LLM, and print the answer.
Summary Diagram
PDF → Chunks → Embeddings → FAISS Index
User Question → Retriever (FAISS) → Relevant Chunks
Prompt (Context + Question) → LLM (Nova Micro) → Answer
Key Points
RAG combines retrieval (FAISS) and generation (LLM) for grounded, context-aware answers.
Amazon Bedrock provides both embeddings and LLM APIs.
LangChain orchestrates the pipeline.
FAISS enables fast similarity search over document chunks.

Summary Diagram
PDF → Chunks → Embeddings → FAISS Index
User Question → Retriever (FAISS) → Relevant Chunks
Prompt (Context + Question) → LLM (Nova Micro) → Answer