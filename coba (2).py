from flask import Flask, request

app = Flask(__name__)

TEXT_FILE = 'data2.txt'  # Update with the actual path to your text file

# Load data from text file with error handling
try:
    with open(TEXT_FILE, 'r', encoding='utf-8') as file:
        documents = file.readlines()

except FileNotFoundError:
    print(f"Error: The file '{TEXT_FILE}' does not exist. Exiting the program or providing a default dataset.")
    # Provide a default dataset or exit the program here if needed.

# Function for text preprocessing (lowercasing)
def preprocess_text(text):
    return text.lower()

# Function to perform search based on a query
def search(query, documents):
    preprocessed_query = preprocess_text(query)

    # Placeholder for document vectors (TF-IDF not implemented)
    document_vectors = [preprocess_text(doc) for doc in documents]

    # Simple keyword matching - check if query is present in documents
    matches = [doc.strip() for doc in document_vectors if preprocessed_query in doc]

    # Return results if found, otherwise return "Not found"
    return matches if matches else ["Not found: Your search did not match any documents."]

# Command-line search
def terminal_search():
    while True:
        query = input("Enter your search query (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        search_results = search(query, documents)
        print("\n".join(search_results))

if __name__ == '__main__':
    terminal_search()
