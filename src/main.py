import os
import requests
import asyncio
import aiohttp
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

class FashionBot:
    def __init__(self):
        self.documents = []
        self.vector_store = None
        self.llm = ChatGroq(temperature=0, groq_api_key=API_KEY, model_name="llama3-70b-8192")
        self.retriever = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation = None
        self.data_fetching = False

    async def scrape_data_from_urls(self, urls):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_content(session, url) for url in urls]
            await asyncio.gather(*tasks)
        self.prepare_vector_store()

    async def fetch_content(self, session, url):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    soup = BeautifulSoup(text, 'html.parser')
                    content = soup.get_text(separator=' ', strip=True)
                    if len(content) > 500:
                        self.documents.append(Document(page_content=content, metadata={"source": url}))
                    else:
                        print(f"Content from {url} is too short to be useful.")
                else:
                    print(f"Failed to retrieve content from {url}, status code: {response.status}")
        except Exception as e:
            print(f"An error occurred while fetching {url}: {e}")

    def prepare_vector_store(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        chunks = text_splitter.split_documents(self.documents)
        
        for i, chunk in enumerate(chunks[:5]):
            print(f"Chunk {i} from {chunk.metadata['source']}:\n{chunk.page_content[:500]}...")
        
        self.vector_store = Chroma.from_documents(chunks, embeddings)
        self.retriever = self.vector_store.as_retriever(k=20)
        self.setup_conversation_chain()
        self.data_fetching = False

    def setup_conversation_chain(self):
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

        self.conversation = ConversationalRetrievalChain.from_llm(
            self.llm, retriever=self.retriever, memory=self.memory, combine_docs_chain_kwargs={"prompt": aqa_prompt}, verbose=True
        )

    def get_urls(self):
        with open('urls.txt', 'r') as file:
            urls = file.read().splitlines()
        return urls

    async def initialize_data(self):
        self.data_fetching = True
        urls = self.get_urls()
        await self.scrape_data_from_urls(urls)

    def get_response(self, question, num_results=20):
        if self.data_fetching:
            return "Currently fetching data, will respond as soon as possible when we have the data."
        try:
            self.retriever.search_kwargs['k'] = num_results
            response = self.conversation.run({'question': question})
            return response
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Sorry, I couldn't generate a response at the moment."




# Get the interval from the .env file
interval = int(os.getenv('DATAFETCH_INTERVAL', 3600))  # Default to 3600 seconds if not set

# Initialize the bot
fashion_bot = FashionBot()

async def periodic_data_fetching(interval):
    while True:
        await fashion_bot.initialize_data()
        await asyncio.sleep(interval)

# Run the periodic data fetching every x seconds
asyncio.run(periodic_data_fetching(interval))

# Expose the fashion_bot instance
def get_fashion_bot():
    return fashion_bot