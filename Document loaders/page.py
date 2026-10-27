from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/macbook-pro/"
data = WebBaseLoader(url)

docs = data.load()
print(len(docs)) # 1 as it is only one web page
print(docs[0].page_content)