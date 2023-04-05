import sqlite3
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List, Tuple, Dict
import tensorflow_hub as hub
from nltk.tokenize import sent_tokenize

# Load the Universal Sentence Encoder model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

def summarize_pages(pages: List[str]) -> List[float]:
    """
    Summarize the given pages using the Universal Sentence Encoder.

    :param pages: A list of strings, where each string represents a page of text.
    :return: A list of float values representing the sparse priming representation.
    """
    # Concatenate the pages into a single string
    text = "\n\n".join(pages)

    # Split the text into sentences
    sentences = sent_tokenize(text)

    # Encode the sentences using the Universal Sentence Encoder
    sentence_embeddings = embed(sentences)

    # Calculate the cosine similarity between each sentence and the others
    similarity_matrix = np.inner(sentence_embeddings, sentence_embeddings)

    # Calculate the average similarity score for each sentence
    sentence_scores = np.mean(similarity_matrix, axis=1)

    # Sort the sentences by score in descending order
    sorted_sentences = sorted(enumerate(sentences), key=lambda x: sentence_scores[x[0]], reverse=True)

    # Keep only the top 10% of the sentences
    top_sentences = sorted_sentences[:int(len(sorted_sentences) * 0.1)]

    # Create a sparse priming representation using the top sentences
    priming = np.zeros(len(sentences))
    for i, sentence in top_sentences:
        priming[i] = 1.0 / len(top_sentences)

    # Return the sparse priming representation
    return priming.tolist()

def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()

def save_data_to_db(embeddings: np.ndarray, metadata: List[str], db_name: str = "embeddings.db"):
    with sqlite3.connect(db_name) as conn:
        pd.DataFrame(embeddings).to_sql("embeddings", conn, if_exists="replace", index=False)
        pd.DataFrame(metadata, columns=["metadata"]).to_sql("metadata", conn, if_exists="replace", index=False)

def load_data(db_name: str = "embeddings.db") -> Tuple[np.ndarray, List[str]]:
    with sqlite3.connect(db_name) as conn:
        embeddings = pd.read_sql("SELECT * FROM embeddings", conn).values
        metadata = pd.read_sql("SELECT * FROM metadata", conn)["metadata"].tolist()
    return embeddings, metadata

def create_index(embeddings: np.ndarray, space: str = "cosinesimil") -> nmslib.dist.FloatIndex:
    index = nmslib.init(space=space)
    index.addDataPointBatch(embeddings)
    index.createIndex()
    return index

def search(query_embedding: np.ndarray, index: nmslib.dist.FloatIndex, metadata: List[str], k: int = 5) -> List[Dict[str, float]]:
    if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 1:
        print("Invalid query_embedding. Please provide a 1D numpy array.")
        return []

    query_embedding = query_embedding.reshape(1, -1)
    nearest_neighbors = index.knnQueryBatch(query_embedding, k=k)[0]
    ids, distances = nearest_neighbors

    results = []
    for i, dist in zip(ids, distances):
        results.append({"metadata": metadata[i], "distance": dist})

    return results

data = load_csv("scientific_papers.csv")

stop_words_set = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words_set]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

data["preprocessed_text"] = data["abstract"].apply(preprocess)

def create_document_embeddings(documents, vector_size=100, window=5, epochs=10, negative=5):
    model = Word2Vec(documents, vector_size=vector_size, window=window, min_count=1, workers=4, epochs=epochs, negative=negative)
    document_vectors = [np.mean([model.wv[word] for word in doc], axis=0) for doc in documents]
    return np.array(document_vectors)

document_embeddings = create_document_embeddings(data["preprocessed_text"])

def cluster_documents(document_embeddings, num_clusters=5, algorithm='kmeans'):
    if algorithm == 'kmeans':
        clustering = KMeans(n_clusters=num_clusters, random_state=42)
    else:
        raise ValueError("Invalid clustering algorithm. Choose 'kmeans', ...")
    return clustering.fit_predict(document_embeddings)

data["cluster"] = cluster_documents(document_embeddings)

def save_data_to_db(data, db_name="data.db"):
    with sqlite3.connect(db_name) as conn:
        data.to_sql("papers", conn, if_exists="replace", index=False)

def load_data_from_db(db_name="papers.db"):
    with sqlite3.connect(db_name) as conn:
        return pd.read_sql("SELECT * FROM papers", conn)

save_data_to_db(data)
data = load_data_from_db()

def analyze_clusters(data, num_clusters=5, return_results=False):
    if not isinstance(data, pd.DataFrame) or "title" not in data.columns or "discipline" not in data.columns or "cluster" not in data.columns:
        print("Invalid input data. Please provide a pandas DataFrame containing 'title', 'discipline', and 'cluster' columns.")
        return
    cluster_analysis = []
    for i in range(num_clusters):
        print(f"Cluster {i + 1}:")
        cluster_data = data[data["cluster"] == i][["title", "discipline"]]
        # Save cluster_data to the database
        save_data_to_db(cluster_data, db_name=f"cluster_{i+1}.db")
        print(cluster_data.to_string(index=False))
        if return_results:
            cluster_analysis.append(cluster_data)
    return cluster_analysis if return_results else None

analyze_clusters(data)
