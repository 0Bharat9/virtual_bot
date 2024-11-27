import subprocess
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
import faiss
import numpy as np

# 1. Load the PDF
pdfreader = PdfReader('Bharat.pdf')

# 2. Extract text from the PDF
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

print(raw_text)

# 3. Split text into smaller chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# 4. Generate embeddings using SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose another embedding model if needed
embeddings = embedding_model.encode(texts)

# 5. Create a FAISS index
embedding_dimension = embeddings.shape[1]  # The dimensionality of the embeddings
index = faiss.IndexFlatL2(embedding_dimension)  # Create a FAISS index for L2 distance
index.add(np.array(embeddings))  # Add the embeddings to the FAISS index

# 6. Function to perform similarity search
def search(query, top_k=5):
    query_embedding = embedding_model.encode([query])  # Embed the query
    distances, indices = index.search(query_embedding, top_k)  # Perform the FAISS search
    return [texts[i] for i in indices[0]]  # Retrieve the most similar chunks of text

# 7. Interactive query loop
while True:
    query = input("Enter your query: ")
    
    # Perform similarity search using FAISS
    docs = search(query)
    
    # Concatenate relevant chunks into context
    context = "\n".join(docs)
    
    # Create the prompt for LLaMA 3.1
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    # 8. Use ollama to generate the answer from LLaMA
    command = ['ollama', 'run', 'llama3.1', prompt]  # Removing --prompt flag and passing prompt directly

    # Open the process and capture the output
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
        for line in process.stdout:
            print(line, end="")  # Print each line as it's received


