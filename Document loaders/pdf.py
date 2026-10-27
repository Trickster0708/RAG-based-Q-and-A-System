from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("Document loaders/GRU.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 10)

chunks = splitter.split_documents(docs)
print(chunks[0].page_content)  # the pdf contains 15 pages so each page is counted as one document
