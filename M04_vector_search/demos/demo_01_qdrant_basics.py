from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, PayloadSchemaType, Filter,
    FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer
from _corpus import get_corpus
from rich.console import Console
from rich.table import Table

console = Console()
COLLECTION = "demo_01_basics"
MODEL_NAME = "intfloat/multilingual-e5-small"

def main() :
    # Chargement du modèle pour créer des embeddings
    # intfloat/multilingual-e5-small : modèle compact multilingue
    console.print(f"Chargement du modèle d'embeddings : {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    console.print(f"Dimension : {dim}\n")

    # Connexion au serveur qdrant local
    client = QdrantClient(url="http://localhost:6333")

    # 1. Créer (ou recréer) la collection
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        console.print("Collection supprimmée")

    # Création de la collection avec la distance cosine
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    console.print(f"Collection {COLLECTION} créée")

    # 2. Ajouter des index payload pour le filtre
    for field in ["product", "version", "category"]:
        client.create_payload_index(COLLECTION, field, PayloadSchemaType.KEYWORD)

    # 3. Indexer le corpus
    corpus = get_corpus()
    console.print(f"Indexation de {len(corpus)} chunks...")

    # Préfixe "passage" : requis par le modèle e5
    textx_to_emb = [f"passage: {c['content']}" for c in corpus]

    embeddings = model.encode(textx_to_emb, normalize_embeddings=True, show_progress_bar=True)

    points = [
        PointStruct(
            id=c["id"],
            vector=embeddings[i].tolist(),
            payload=c
        )
        for i, c in enumerate(corpus)
    ]

    # Insertion des points dans Qdrant
    client.upsert(collection_name=COLLECTION, points=points, wait=True)

    console.print(f"{len(points)} chunks indexés\n")

if __name__ == "__main__":
    main()