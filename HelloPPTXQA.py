from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

#import os
#os.environ['OPENAI_API_KEY']="sk-XXXXXXXXXXXXXXXXXXX"

from langchain.document_loaders import TextLoader

loader = UnstructuredPowerPointLoader("data/Sample.pptx")
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(pages, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))

query = "どんなタスクがありますか？納品日はいつですか？"
result = qa.run(query)
print(result)
