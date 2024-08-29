import os
import requests
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv

load_dotenv()

# Load API Key
API_KEY = os.getenv("GROQ_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Function to scrape data from URLs
"""
Function to scrape data from a list of URLs.

This function takes a list of URLs, sends HTTP GET requests to each URL, and retrieves the HTML content. 
It then parses the HTML content using BeautifulSoup to extract the text. If the extracted text is substantial 
(more than 500 characters), it creates a Document object with the text and the URL as metadata and adds it to 
a list of documents. If the content is too short or the request fails, it prints an appropriate message.

Args:
    urls (list): A list of URLs to scrape data from.

Returns:
    list: A list of Document objects containing the scraped text and metadata.
"""
def scrape_data_from_urls(urls):
    documents = []
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            if len(text) > 500:  # Ensure the content is substantial
                documents.append(Document(page_content=text, metadata={"source": url}))
            else:
                print(f"Content from {url} is too short to be useful.")
        else:
            print(f"Failed to retrieve content from {url}, status code: {response.status_code}")
    return documents

# Function to prepare vector store with Chroma
def prepare_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    chunks = text_splitter.split_documents(documents)
    
    # Debugging: Print some of the chunks to verify
    for i, chunk in enumerate(chunks[:5]):  # Print first 5 chunks
        print(f"Chunk {i} from {chunk.metadata['source']}:\n{chunk.page_content[:500]}...")  # Debugging: print first 500 chars of chunk
    
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

# Initialize the language model with Groq API
llm = ChatGroq(temperature=0, groq_api_key=API_KEY, model_name="llama3-70b-8192")




# List of URLs to scrape data from
def get_urls():
    return [
        'https://www.khaadi.com/',
        'https://pk.sapphireonline.pk/',
        'https://generation.com.pk/',
        'https://myrangja.com/collections/new-arrivals'
    ]


# Scrape and prepare documents
documents = scrape_data_from_urls(get_urls())
print("Scraped Documents:")
for doc in documents[:5]:  # Print the first 5 documents
    print(f"Source: {doc.metadata['source']}, Content: {doc.page_content[:200]}")  # Print source and first 200 characters

# Prepare vector store with all documents
vector_store = prepare_vector_store(documents)
print("Vector store prepared with documents from both sources.")

# Adjust the retrieval setup
retriever = vector_store.as_retriever(k=20)  # Increase k to retrieve more chunks
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prepare the prompt to reference the source of each item
general_system_template = """
You are a helpful assistant. When responding to questions, provide the relevant items from all brands, and make sure to clearly mention which brand each item is from.

----
chat history = {chat_history}
----
context = {context}
----
human question =  {question}
----
Provide results, specifying the brand for each item (e.g., "This item is from Sapphire").
"""

general_user_template = "Question:```{question}```"

messages = [
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template(general_user_template)
]
aqa_prompt = ChatPromptTemplate.from_messages(messages)

# Configure the conversation chain to include more results
conversation = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={"prompt": aqa_prompt}, verbose=True
)

# Function to get responses from the conversation chain
def get_response(question, num_results=20):
    try:
        retriever.search_kwargs['k'] = num_results  # Set the number of results in retriever
        response = conversation.run({'question': question})
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Sorry, I couldn't generate a response at the moment."
