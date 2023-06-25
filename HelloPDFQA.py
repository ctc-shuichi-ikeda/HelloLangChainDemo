from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

#import os
#os.environ['OPENAI_API_KEY']="sk-XXXXXXXXXXXXXXXXXXX"

from langchain.document_loaders import TextLoader

loader = PyPDFLoader("data/Sample.pdf")
pages = loader.load_and_split()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(pages[0])

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(pages, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))

query = "日付はいつですか？"
result = qa.run(query)
print(result)
