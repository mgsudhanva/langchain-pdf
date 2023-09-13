from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

OPENAI_API_KEY = ""

loader = UnstructuredPDFLoader("./paper.pdf")

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pinecone.init(      
	api_key='f3ad5b01-3721-4296-940c-2350dbf75bce',      
	environment='gcp-starter'      
)      
index = pinecone.Index('qa')

vector_store = Pinecone(index, embeddings.embed_query, "text")

qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), chain_type="stuff", retriever=vector_store.as_retriever())

print("Connector development help bot. What do you want to know?")
while True:
    query = input("")
    answer = qa.run(query)
    print(answer)
    print("\nWhat else can I help you with:")
