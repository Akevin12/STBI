from flask import Flask, render_template, request
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import os
import re

app = Flask(__name__)

# load stop words
stop_words = set(stopwords.words('english'))

# load WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# load Porter Stemmer
ps = PorterStemmer()

# remove punctuation, stop words, tokenization, stemming, and lemmatization
def preprocess_text(text):
    # remove punctuation
    text = re.sub(r'[.,!$(*)%@]', '', text)
    # lowercasing
    text = text.lower()
    # tokenization
    tokens = word_tokenize(text)
    # stop words removal
    tokens = [token for token in tokens if token not in stop_words]
    # Stemming and Lemmatization
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Handling data set
SPORT_FOLDER = 'sport'

try:
    documents = []
    filenames = []

    for filename in os.listdir(SPORT_FOLDER):
        file_path = os.path.join(SPORT_FOLDER, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            documents.append(content)
            filenames.append(filename)

except FileNotFoundError:
    print(f"Error: The folder '{SPORT_FOLDER}' does not exist")

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    similarity = dot_product / (norm_vec1 * norm_vec2 + 1e-8)  # hindarin pembagian 0

    return similarity

def calculate_tfidf(documents): 
    # get kumpulan kata unik dari semua dokumen setelah melalui tahap preprocessing
    unique_words = sorted(set(word for doc in documents for word in preprocess_text(doc).split()))
    # semacam dicitonary buat index (dipetakan gitu)
    word_indices = {word: i for i, word in enumerate(unique_words)}

    # hitung matriks TF-IDF dengan mengisi frekuensi kemunculan kata dan menghitung frekuensi dokumen.
    tfidf_matrix = np.zeros((len(documents), len(unique_words)), dtype=float)

    for i, doc in enumerate(documents):
        preprocessed_doc = preprocess_text(doc)
        for word in preprocessed_doc.split():
            if word in word_indices:
                tfidf_matrix[i, word_indices[word]] += 1

    # hitung frek dokumen
    document_frequency = np.sum(tfidf_matrix > 0, axis=0)
    # hitung total idf untuk setiap kata
    idf = np.log(len(documents) / (document_frequency + 1))

    tfidf_matrix = tfidf_matrix * idf

    tfidf_norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)

    # normalisasi matriks dengan membagi setiap barisnya dengan norma euclidean buat
    # mengukur seberapa panjang vektor yang merepresentasikan setiap dokumen 
    tfidf_matrix_normalized = np.divide(tfidf_matrix, tfidf_norms, where=tfidf_norms != 0)

    return tfidf_matrix_normalized, unique_words

tfidf_matrix, unique_words = calculate_tfidf(documents)

def search(query, tfidf_matrix, file_document_pairs, n_results):
    if not query:
        return ["Please enter a search query."], False

    preprocessed_query = preprocess_text(query)
    query_vector = np.zeros((1, len(unique_words)), dtype=float)

    for word in preprocessed_query.split():
        for unique_word in unique_words:
            if unique_word.startswith(word):
                query_vector[0, unique_words.index(unique_word)] += 1

    query_vector = query_vector.flatten()
    query_vector = query_vector / np.linalg.norm(query_vector)

    tfidf_norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    tfidf_matrix_normalized = np.divide(tfidf_matrix, tfidf_norms, where=tfidf_norms != 0)

    # hitung similarity
    similarities = np.dot(query_vector, tfidf_matrix_normalized.T)

    # handle NaN and zero similarity
    valid_indices = np.where(~np.isnan(similarities) & (similarities > 0))[0]
    
    if not valid_indices.any():
        return [], False

    top_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]][:n_results]

    matches = []
    for i in top_indices:
        filename, document = file_document_pairs[i]
        matches.append((filename, document, similarities[i]))

    matches.sort(key=lambda x: x[2], reverse=True)

    return matches, bool(matches)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_results():
    if request.method == 'POST':
        query = request.form['query']

        # Combine filenames and documents into a list of tuples
        file_document_pairs = list(zip(filenames, documents))

        matches, success = search(query, tfidf_matrix, file_document_pairs, 1000)

    result_data = []
    for filename, document, similarity in matches:
        result_data.append({
        'filename': filename,
        'document': document[:10000],
        'similarity': similarity
    })

    return render_template('result.html', search_results=result_data, success=success)
    
if __name__ == '__main__':
    app.run(debug=True)
