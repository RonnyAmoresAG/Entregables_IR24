import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
from nltk.corpus import stopwords

# Step 1: Import Libraries (Ya está hecho)
# import pandas as pd
# import stringg

# Step 2: Load the Dataset
df = pd.read_csv('data/podcastdata_dataset.csv')
print(df.head())

# Step 3: Text Preprocessing
corpus = df['text']

# Eliminar puntuación
corpus_nopunct = [doc.lower().translate(str.maketrans('', '', string.punctuation)) for doc in corpus]

# Eliminar palabras de parada
stopw = set(stopwords.words('english'))
corpus_nostopw = [' '.join([word for word in doc.split() if word not in stopw]) for doc in corpus_nopunct]

df['text_nostopw'] = corpus_nostopw
print(df.head())

# Step 4: Vector Space Representation - TF-IDF
vectorizer = TfidfVectorizer()
tfidf_mtx = vectorizer.fit_transform(df['text_nostopw'])

# Step 5: Vector Space Representation - BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

def generate_bert_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :])  # Use [CLS] token representation
    return np.array(embeddings).squeeze()

corpus_bert = generate_bert_embeddings(corpus_nostopw[:50])

# Step 6: Query Processing
def retrieve_tfidf(query):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(tfidf_mtx, query_vector)
    similarities_df = pd.DataFrame(similarities, columns=['sim'])
    similarities_df['ep'] = df['title']
    return similarities_df

def retrieve_bert(query):
    query_bert = generate_bert_embeddings(query)
    similarities = cosine_similarity(corpus_bert.reshape(50, 768), query_bert.reshape(1, 768))
    similarities_df = pd.DataFrame(similarities, columns=['sim'])
    similarities_df['ep'] = df['title']
    return similarities_df

# Step 7: Retrieve and Compare Results
query = 'Computer Science'
tfidf_results = retrieve_tfidf(query)
bert_results = retrieve_bert([query])

print("TF-IDF Results:\n", tfidf_results.sort_values(by='sim', ascending=False).head(10))
print("BERT Results:\n", bert_results.sort_values(by='sim', ascending=False).head(10))

# Step 8: Test the IR System
sample_query = 'Artificial Intelligence'
tfidf_results_sample = retrieve_tfidf(sample_query)
bert_results_sample = retrieve_bert([sample_query])

print("TF-IDF Sample Query Results:\n", tfidf_results_sample.sort_values(by='sim', ascending=False).head(10))
print("BERT Sample Query Results:\n", bert_results_sample.sort_values(by='sim', ascending=False).head(10))

# Step 9: Compare Results
# Analyze and compare the results obtained from TF-IDF and BERT representations.
# Discuss the differences, strengths, and weaknesses of each method based on the retrieval results.
