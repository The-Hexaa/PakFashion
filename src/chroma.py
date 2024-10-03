

from chromadb import Client

# Connect to ChromaDB
client = Client()

# Specify your collection name
collection_name = 'clothing_brands'  # Replace with your actual collection name

# Get the collection object
collection = client.get_collection(collection_name)

# Fetch documents from the collection
documents = collection.get()  # Fetch all documents in the collection

# Print the fetched documents
print(documents)
