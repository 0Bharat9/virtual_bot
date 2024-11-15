from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from typing_extensions import Concatenate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


os.environ["OPENAI_API_KEY"] = "KEYTOUSEFAISSGPU" #note for bharat in order to use this you need openapi key which will be around 200-300rs

pdfreader = PdfReader('/Users/puneetwalia/Desktop/minorpra/B (1)_merged_merged.pdf')

raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

print(raw_text)

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)
embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts(texts, embeddings)
chain = load_qa_chain(OpenAI(), chain_type="stuff")
while True:
  query = input("Enter your query: ")
  docs = document_search.similarity_search(query)
  y=chain.run(input_documents=docs, question=query)
  print(y)
