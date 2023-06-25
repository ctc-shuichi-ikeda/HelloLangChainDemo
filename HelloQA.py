from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os
os.environ['OPENAI_API_KEY']="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"

from langchain.document_loaders import TextLoader
loader = TextLoader("data/Message.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))

query = "あなたの趣味は何ですか？"
result = qa.run(query)
print(result)
