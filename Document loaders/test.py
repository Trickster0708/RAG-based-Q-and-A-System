from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
splitter = CharacterTextSplitter(separator = "", chunk_size = 10, chunk_overlap = 1)
# with the help of separator, doc is divided into given chunks

data = TextLoader("Document loaders/notes.txt")
docs =data.load()

chunks = splitter.split_documents(docs)

print(len(chunks))

for i in chunks:
    print(i.page_content)
    print()
    print()
