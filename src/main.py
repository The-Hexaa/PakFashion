import logging
import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from urls_finder import URLFinder  # Import URLFinder

class RAGWithGroq:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Load environment variables
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.search_query = os.getenv("SEARCH_QUERY", "Pakistani women clothing brands")

        # Use URLFinder to access the ChromaDB client and collection
        url_finder = URLFinder()
        self.collection = url_finder.collection  # Access the collection directly

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose a suitable model

    def generate_query_embedding(self, query):
        """Generates embedding for the given query using SentenceTransformer."""
        return self.embedding_model.encode(query).tolist()

    def retrieve_from_chroma(self, query_embedding):
        """Retrieves relevant documents from ChromaDB using the query embedding."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=5  # Adjust number of results as needed
        )
        # Check if results contain any data
        if not results or not results['documents'][0]:
            self.logger.error("No relevant documents found in ChromaDB.")
            return None
        
        self.logger.debug(f"Retrieved results from Chroma: {results}")
        return results

    def groq_generate(self, query, context):
        """Generates a response using Groq API based on the query and retrieved context."""
        url = "https://api.groq.com/v1/complete"  # Verify this URL
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": f"{context}\n\nUser query: {query}\n",
            "max_tokens": 150
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raise an error for bad responses
            return response.json().get('text')
        except requests.exceptions.HTTPError as http_err:
            self.logger.error(f"HTTP error occurred: {http_err}")
            return "Failed to generate a response from Groq due to an HTTP error."
        except Exception as err:
            self.logger.error(f"An error occurred: {err}")
            return "An unexpected error occurred while generating a response."

    def rag_pipeline(self, query):
        """RAG pipeline: retrieve documents from ChromaDB and generate a response using Groq."""
        self.logger.info(f"Processing query: {query}")

        # Step 1: Generate embedding for the query
        query_embedding = self.generate_query_embedding(query)

        # Step 2: Retrieve relevant data from ChromaDB
        retrieved_data = self.retrieve_from_chroma(query_embedding)
        
        if retrieved_data is None:
            self.logger.info("No relevant data found. Unable to generate response.")
            return "No relevant data found in the database."
        
        # Extract context from retrieved data
        context = " ".join([doc.get("description", "No description available") for doc in retrieved_data['metadatas']])
        self.logger.info(f"Retrieved context: {context}")

        # Step 3: Send the context and query to Groq for generation
        response = self.groq_generate(query, context)
        self.logger.info(f"Generated response: {response}")

        return response

if __name__ == "__main__":
    rag_groq = RAGWithGroq()
    user_query = "Can you recommend Pakistani clothing brands?"
    response = rag_groq.rag_pipeline(user_query)
    print(f"Response: {response}")
