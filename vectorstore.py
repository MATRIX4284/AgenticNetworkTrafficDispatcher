def load_sop(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    


from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def split_sop(text: str):
    splitter = MarkdownTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_text(text)



# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_community.vectorstores import Chroma

sop_text = load_sop("sop_routing.md")
chunks = split_sop(sop_text)

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="telecom_sop",
    persist_directory="./chroma_sop"
)

vectorstore.persist()

def get_vectorstore():
    return vectorstore

print("✅ SOP embedded into Chroma")