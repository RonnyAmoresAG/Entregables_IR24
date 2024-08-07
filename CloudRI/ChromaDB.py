# Importar el cliente de ChromaDB
import chromadb

# Crear un cliente ChromaDB
chroma_client = chromadb.Client()

# Crear una colección llamada "my_collection"
collection = chroma_client.create_collection(name="my_collection")

# Importar la biblioteca pandas para manejar el DataFrame
import pandas as pd

# Cargar el DataFrame desde el archivo CSV
wine_df = pd.read_csv("data/winemag-data-130k-v2.csv")

# Extraer las primeras 30 descripciones de los vinos
documents = wine_df['description'][:30].tolist()

# Extraer los primeros 30 IDs de los vinos y convertirlos a strings
ids = wine_df['Unnamed: 0'][:30].astype(str).tolist()

# Insertar los 30 documentos en la colección "my_collection"
collection.add(
    documents=documents,  # Lista de descripciones de los vinos
    ids=ids  # Lista de IDs convertidos a strings
)

# Realizar una consulta en la colección con un texto de prueba
results = collection.query(
    query_texts=["This is a query document about florida"],  # Texto de consulta
    n_results=2  # Número de resultados a retornar
)

# Imprimir los resultados de la primera consulta
print(results)

# Realizar otra consulta en la colección con un texto diferente
query_text = "This is a query document about wine"
results = collection.query(
    query_texts=[query_text],  # Texto de consulta
    n_results=5  # Número de resultados a retornar
)

# Imprimir los resultados de la segunda consulta
print(results)

# Recorrer los resultados para imprimir los IDs y las descripciones de los documentos más similares
for doc_id, doc_text in zip(results['ids'][0], results['documents'][0]):
    print(f"Document ID: {doc_id}")  # Imprimir el ID del documento
    print(f"Document Text: {doc_text}\n")  # Imprimir el texto del documento
