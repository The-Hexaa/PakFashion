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
        if not results or not results.get('documents'):
            self.logger.error("No relevant documents found in ChromaDB.")
            return None
        
        self.logger.debug(f"Retrieved results from Chroma: {results}")
        return results

    def groq_generate(self, query, context):
        """Generates a response using Groq API based on the query and retrieved context."""
        url = "https://api.groq.com/v1/complete"  # Ensure this is the correct Groq API URL
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Creating the prompt by combining the user's query with the retrieved context from ChromaDB
        prompt = f"Query: {query}\n\nContext:\n{context}\n\nGenerate a detailed response based on the query and the context."

        payload = {
            "prompt": prompt,
            "max_tokens": 300,  # Adjust the token count based on your response length requirement
            "temperature": 0.7,  # Adjust this to control randomness in generation
        }

        try:
            # Make a POST request to the Groq API with the prompt
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()

            # Return the generated response from Groq
            generated_response = data.get("choices", [{}])[0].get("text", "No response generated")
            self.logger.info(f"Generated response from Groq: {generated_response}")
            return generated_response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error occurred during Groq API request: {e}")
            return "An error occurred while generating a response."



    def rag_pipeline(self, user_input):
        # Assuming you have a method to get embeddings from user input
        user_embedding = self.embedding_model.encode(user_input).tolist()

        # Fetch relevant documents based on the user input
        retrieved_data = self.collection.query(query_embeddings=[user_embedding], n_results=5)

        # Log the retrieved data for debugging
        self.logger.debug(f"Retrieved data: {retrieved_data}")

        # Use a safe method to extract descriptions
        context = " ".join(
            [doc.get("description", "No description available") for doc in retrieved_data['metadatas'] if doc]
        )

def get_fashion_bot():
    """Function to instantiate and return the RAGWithGroq object."""
    return RAGWithGroq()
