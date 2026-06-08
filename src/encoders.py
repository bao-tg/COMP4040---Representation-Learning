import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
import torch
import os
from tqdm import tqdm

class BaseEncoder:
    def encode(self, texts):
        raise NotImplementedError("Subclasses must implement encode method")

class TFIDFEncoder(BaseEncoder):
    def __init__(self, n_components=300):
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.svd = TruncatedSVD(n_components=n_components)
        self.is_fitted = False

    def fit(self, texts):
        print("Fitting TF-IDF + SVD...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf_matrix)
        self.is_fitted = True

    def encode(self, texts):
        if not self.is_fitted:
            self.fit(texts)
        tfidf_matrix = self.vectorizer.transform(texts)
        return self.svd.transform(tfidf_matrix)

class W2VEncoder(BaseEncoder):
    def __init__(
        self,
        pretrained_path="GoogleNews-vectors-negative300.bin",
        pretrained_name="word2vec-google-news-300",
        vector_size=300,
        allow_download=False,
    ):
        self.model = None
        self.vector_size = vector_size
        self.pretrained_path = pretrained_path
        self.pretrained_name = pretrained_name
        if pretrained_path and not os.path.exists(pretrained_path) and not pretrained_path.endswith(".gz"):
            gz_fallback = pretrained_path + ".gz"
            if os.path.exists(gz_fallback):
                pretrained_path = gz_fallback
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained(pretrained_path)
        elif allow_download:
            self.load_from_gensim(pretrained_name)

    def load_pretrained(self, pretrained_path):
        print(f"Loading pretrained Word2Vec vectors from {pretrained_path}...")
        self.model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
        self.vector_size = int(self.model.vector_size)

    def load_from_gensim(self, pretrained_name):
        import gensim.downloader as api

        print(f"Loading pretrained Word2Vec vectors from gensim-data:{pretrained_name}...")
        self.model = api.load(pretrained_name)
        self.vector_size = int(self.model.vector_size)

    def encode(self, tokenized_texts):
        if self.model is None:
            raise FileNotFoundError(
                "Missing pretrained Word2Vec vectors. Place GoogleNews-vectors-negative300.bin.gz "
                "or GoogleNews-vectors-negative300.bin at the repo root, "
                "or instantiate W2VEncoder(allow_download=True)."
            )
        
        embeddings = []
        for tokens in tqdm(tokenized_texts, desc="W2V Encoding"):
            vectors = [self.model[word] for word in tokens if word in self.model]
            if vectors:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings)

class GloVeEncoder(BaseEncoder):
    def __init__(self, glove_path=None, vector_size=300):
        self.embeddings_index = {}
        self.embeddings = self.embeddings_index
        self.vector_size = vector_size
        if glove_path and os.path.exists(glove_path):
            self.load_glove(glove_path)

    def load_glove(self, glove_path):
        print(f"Loading GloVe vectors from {glove_path}...")
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading GloVe"):
                values = line.split()
                if len(values) < self.vector_size + 1:
                    continue
                word = " ".join(values[:-self.vector_size])
                coefs = np.asarray(values[-self.vector_size:], dtype='float32')
                self.embeddings_index[word] = coefs
        print(f"Loaded {len(self.embeddings_index)} GloVe vectors.")

    def encode(self, tokenized_texts):
        embeddings = []
        for tokens in tqdm(tokenized_texts, desc="GloVe Encoding"):
            vectors = [self.embeddings_index[word] for word in tokens if word in self.embeddings_index]
            if vectors:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings)

class SBERTEncoder(BaseEncoder):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading SBERT model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts, batch_size=256):
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

class BGEEncoder(BaseEncoder):
    def __init__(self, model_name='BAAI/bge-large-en-v1.5'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading BGE model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts, batch_size=256):
        # BGE models usually recommend a prompt for retrieval, 
        # but for general representation we use the text as is.
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
