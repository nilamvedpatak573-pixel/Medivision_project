from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# import your document loading code here
from langchain_community.document_loaders import PyPDFLoader

# 1. Load your medical document
loader = PyPDFLoader("data/medibot data/Current Essentials of Medicine.pdf")

documents = loader.load()

# 2. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Create FAISS database
vectorstore = FAISS.from_documents(
    documents,
    embeddings
)

# 4. Save FAISS database
vectorstore.save_local(
    "vectorstore/db_faiss"
)

print("FAISS vectorstore created successfully!")