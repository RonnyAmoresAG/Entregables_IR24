import numpy as np
from multiprocessing import Pool, cpu_count, set_start_method
import pandas as pd
from transformers import BertTokenizer, TFBertModel
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import os
import tensorflow as tf

# Configurar TensorFlow para desactivar el paralelismo interno
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Usar todos menos dos núcleos
num_cpus = max(1, cpu_count() - 2)

# Variable global para el modelo
word2vec_model = None

def initialize_word2vec_model():
    global word2vec_model
    word2vec_model = api.load('word2vec-google-news-300')

# Definir funciones para generar embeddings
def generate_word2vec_embedding(text):
    tokens = text.lower().split()
    word_vectors = [word2vec_model[word] for word in tokens if word in word2vec_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def generate_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

# Funciones para paralelizar embeddings
def parallel_generate_word2vec_embeddings(texts):
    with Pool(num_cpus, initializer=initialize_word2vec_model) as pool:
        embeddings = pool.map(generate_word2vec_embedding, texts)
    return np.array(embeddings)

def parallel_generate_bert_embeddings(texts):
    from multiprocessing.pool import ThreadPool
    with ThreadPool(num_cpus) as pool:
        embeddings = pool.map(generate_bert_embedding, texts)
    return np.array(embeddings)

# Función para dividir el corpus en partes más pequeñas
def divide_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Cargar modelos preentrenados
    print("Cargando modelo Word2Vec...")
    initialize_word2vec_model()
    print("Modelo Word2Vec cargado.")

    print("Cargando modelo BERT...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')
    print("Modelo BERT cargado.")

    # Cargar el dataset
    print("Cargando el dataset...")
    wine_df = pd.read_csv('data/winemag-data_first150k.csv')
    corpus = wine_df['description'].tolist()
    print(f"Dataset cargado con {len(corpus)} descripciones.")

    # Dividir el corpus en partes más grandes
    print("Dividiendo el corpus en partes más grandes...")
    chunk_size = len(corpus) // num_cpus  # Ajustar el número de chunks
    corpus_chunks = list(divide_chunks(corpus, chunk_size))
    print(f"Corpus dividido en {len(corpus_chunks)} partes.")

    # Definir rutas para guardar los embeddings y las similitudes
    word2vec_embeddings_path = 'corpus_word2vec_embeddings.npy'
    bert_embeddings_path = 'corpus_bert_embeddings.npy'
    word2vec_similarity_path = 'corpus_word2vec_similarity.npy'
    bert_similarity_path = 'corpus_bert_similarity.npy'

    # Generar o cargar embeddings de Word2Vec
    if os.path.exists(word2vec_embeddings_path):
        print("Cargando embeddings de Word2Vec desde disco...")
        corpus_word2vec_embeddings = np.load(word2vec_embeddings_path)
        print("Embeddings de Word2Vec cargados.")
    else:
        print("Generando embeddings de Word2Vec...")
        word2vec_embeddings_list = []
        for i, chunk in enumerate(corpus_chunks):
            print(f"Procesando chunk {i+1}/{len(corpus_chunks)} de Word2Vec...")
            word2vec_embeddings = parallel_generate_word2vec_embeddings(chunk)
            word2vec_embeddings_list.append(word2vec_embeddings)
        corpus_word2vec_embeddings = np.vstack(word2vec_embeddings_list)
        np.save(word2vec_embeddings_path, corpus_word2vec_embeddings)
        print("Embeddings de Word2Vec generados y guardados en disco.")

    # Generar o cargar embeddings de BERT
    if os.path.exists(bert_embeddings_path):
        print("Cargando embeddings de BERT desde disco...")
        corpus_bert_embeddings = np.load(bert_embeddings_path)
        print("Embeddings de BERT cargados.")
    else:
        print("Generando embeddings de BERT...")
        bert_embeddings_list = []
        for i, chunk in enumerate(corpus_chunks):
            print(f"Procesando chunk {i+1}/{len(corpus_chunks)} de BERT...")
            bert_embeddings = parallel_generate_bert_embeddings(chunk)
            bert_embeddings_list.append(bert_embeddings)
        corpus_bert_embeddings = np.vstack(bert_embeddings_list)
        np.save(bert_embeddings_path, corpus_bert_embeddings)
        print("Embeddings de BERT generados y guardados en disco.")

    # Calcular o cargar la similitud entre embeddings de Word2Vec
    if os.path.exists(word2vec_similarity_path):
        print("Cargando similitudes de Word2Vec desde disco...")
        corpus_word2vec_similarity = np.load(word2vec_similarity_path)
        print("Similitudes de Word2Vec cargadas.")
    else:
        print("Calculando la similitud entre embeddings de Word2Vec...")
        corpus_word2vec_similarity = cosine_similarity(corpus_word2vec_embeddings)
        np.save(word2vec_similarity_path, corpus_word2vec_similarity)
        print("Similitud entre embeddings de Word2Vec calculada y guardada en disco.")

    # Calcular o cargar la similitud entre embeddings de BERT
    if os.path.exists(bert_similarity_path):
        print("Cargando similitudes de BERT desde disco...")
        corpus_bert_similarity = np.load(bert_similarity_path)
        print("Similitudes de BERT cargadas.")
    else:
        print("Calculando la similitud entre embeddings de BERT...")
        corpus_bert_similarity = cosine_similarity(corpus_bert_embeddings)
        np.save(bert_similarity_path, corpus_bert_similarity)
        print("Similitud entre embeddings de BERT calculada y guardada en disco.")

    # Generar embeddings para la consulta
    query = "A red wine with steak"
    print("Generando embeddings para la consulta...")
    query_word2vec_embedding = generate_word2vec_embedding(query)
    query_bert_embedding = generate_bert_embedding(query).reshape(1, -1)
    print("Embeddings para la consulta generados.")

    # Calcular la similitud entre la consulta y los documentos
    print("Calculando la similitud entre la consulta y los documentos con Word2Vec...")
    query_word2vec_similarity = cosine_similarity(query_word2vec_embedding.reshape(1, -1), corpus_word2vec_embeddings)
    print("Similitud con Word2Vec calculada.")

    print("Calculando la similitud entre la consulta y los documentos con BERT...")
    query_bert_similarity = cosine_similarity(query_bert_embedding, corpus_bert_embeddings)
    print("Similitud con BERT calculada.")

    # Recuperar y clasificar documentos basados en los puntajes de similitud
    def retrieve_and_rank_documents(similarity_scores, top_n=10):
        sorted_indices = np.argsort(similarity_scores[0])[::-1]
        top_indices = sorted_indices[:top_n]
        return wine_df.iloc[top_indices]

    print("Recuperando y clasificando documentos basados en los puntajes de similitud...")
    top_word2vec_results = retrieve_and_rank_documents(query_word2vec_similarity)
    top_bert_results = retrieve_and_rank_documents(query_bert_similarity)

    print("Top Word2Vec Results:\n", top_word2vec_results[['description', 'points', 'price']])
    print("Top BERT Results:\n", top_bert_results[['description', 'points', 'price']])
