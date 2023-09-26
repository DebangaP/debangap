!pip install -q llama-index
!pip install llama-cpp-python
!pip install transformers
!pip install accelerate
!pip install pinecone-client


from llama_index.vector_stores import PineconeVectorStore
from llama_index import StorageContext, VectorStoreIndex, SimpleDirectoryReader, download_loader
from llama_index import ServiceContext, set_global_service_context
import pinecone
from IPython.display import Markdown, display
from pathlib import Path

#llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)

# configure service context
service_context = ServiceContext.from_defaults(chunk_size=50, chunk_overlap=10)

PDFReader = download_loader("PDFReader")
loader = PDFReader()
documents = SimpleDirectoryReader("PDFs").load_data

documents = loader.load_data(file=Path('./PDFs/HR_Policy.pdf'))

# init pinecone
pinecone.init(api_key="6", environment="gcp-starter")
pinecone.create_index("hr-index", dimension=384, metric="euclidean", pod_type="p1")

# construct vector store and customize storage context
storage_context = StorageContext.from_defaults(
    vector_store=PineconeVectorStore(pinecone.Index("quickstart"))
)
pinecone_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

in_memory_vector_index = VectorStoreIndex.from_documents(documents)
in_memory_vector_index.storage_context.persist()

query_engine = in_memory_vector_index.as_query_engine()
response = query_engine.query("What is the objective of the HR Policy? ")

display(Markdown(f"<b>{response}</b>"))
