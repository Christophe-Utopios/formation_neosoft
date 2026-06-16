from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel

console = Console()
COLLECTION = "demo_05_parent_doc"
MODEL_NAME = "intfloat/multilingual-e5-small"


# Document long (un guide complet, pas un FAQ chunk)
LONG_DOCUMENT = """# Configuration LDAP de NovaCloud

## Introduction
NovaCloud supporte l'authentification via un annuaire LDAP externe (Active Directory ou OpenLDAP).
Cette section explique la configuration complète, les prérequis, et les étapes de dépannage.

## Prérequis
Avant de commencer, vous devez disposer des éléments suivants :
- L'URL du serveur LDAP, sous la forme ldap://serveur ou ldaps://serveur pour la version chiffrée.
- Le port d'écoute, qui est par défaut 389 pour LDAP non chiffré et 636 pour LDAPS.
- Un compte de service avec un Bind DN qui aura les permissions de lecture sur l'annuaire.
- Le Base DN qui définit la racine de recherche pour les utilisateurs.
- Si vous utilisez LDAPS avec un certificat auto-signé, le certificat de l'autorité de certification.

## Configuration via l'interface graphique
Connectez-vous à NovaCloud avec un compte administrateur. Ouvrez le menu Paramètres,
puis allez dans la section Authentification, puis sous-section LDAP.
Activez la bascule "Authentification LDAP". Renseignez les champs URL serveur, Bind DN,
mot de passe associé, et Base DN. Cliquez sur "Tester la connexion" : le système doit retourner
"OK - 1543 utilisateurs trouvés" ou un nombre approprié à votre annuaire.

## Mappage des attributs
NovaCloud doit savoir où chercher chaque information utilisateur dans l'annuaire LDAP.
Le mappage par défaut est : email → mail, prénom → givenName, nom → sn, groupes → memberOf.
Vous pouvez personnaliser ces correspondances si votre annuaire utilise des attributs custom.
Pour les groupes, le filtre par défaut récupère tous les memberOf de l'utilisateur.

## Synchronisation périodique
Par défaut, NovaCloud synchronise les utilisateurs LDAP toutes les heures.
La fréquence est paramétrable entre 5 minutes et 24 heures.
Une synchronisation manuelle est possible via le bouton "Synchroniser maintenant".
Un journal des synchronisations est disponible dans Paramètres > Logs > Synchronisation.

## Dépannage
Si la connexion échoue, plusieurs causes possibles. Premièrement vérifiez les logs côté NovaCloud
dans /var/log/novacloud/auth.log. Les erreurs courantes sont :
- "Bind failed" indique que le compte de service est invalide ou son mot de passe a changé.
- "No such object" indique que le Base DN configuré ne correspond à rien dans l'annuaire.
- "TLS handshake failed" indique un problème de certificat avec LDAPS.

Pour les certificats auto-signés ou des CA internes, vous devez importer le certificat de l'autorité
dans le truststore de NovaCloud via la commande :
    novacloud-cli ldap import-ca <chemin/vers/ca.pem>
Après import, redémarrez le service d'authentification.
"""


def main() -> None:
    console.print("\n[bold orange3]Demo 05 — Parent-document retrieval[/bold orange3]\n")

    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(url="http://localhost:6333")

    # 1. Découpage à deux niveaux : parents (sections) et enfants (petits chunks)
    # Parents : grands chunks de ~600 caractères (sections logiques)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=0,
        separators=["\n## ", "\n# "],  # Découpe sur les titres de sections
    )
    parents = parent_splitter.split_text(LONG_DOCUMENT)
    console.print(f"[cyan]→ Document découpé en {len(parents)} parents (sections)[/cyan]\n")

    # Enfants : petits chunks de ~200 caractères pour la recherche précise
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=["\n", ". ", " "],
    )

    # Création des embeddings : on indexe les enfants mais on garde le parent en payload
    points: list[PointStruct] = []
    point_id = 0
    for parent_id, parent in enumerate(parents):
        children = child_splitter.split_text(parent)
        for child in children:
            point_id += 1
            # L'embedding est créé sur le petit chunk (précis)
            child_emb = model.encode(f"passage: {child}", normalize_embeddings=True)
            points.append(PointStruct(
                id=point_id,
                vector=child_emb.tolist(),
                payload={
                    "child_content": child,
                    "parent_id": parent_id,
                    "parent_content": parent,  # On stocke le contexte complet
                },
            ))

    console.print(f"[cyan]→ {len(points)} petits chunks (children) générés[/cyan]")
    console.print(f"[cyan]→ Chaque petit chunk garde une référence vers son parent[/cyan]\n")

    # 2. Indexation dans Qdrant
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    dim = model.get_sentence_embedding_dimension()
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    client.upsert(COLLECTION, points=points, wait=True)

    # 3. Rechercher
    query = "Comment importer un certificat CA pour LDAPS ?"
    console.print(f"[bold cyan]Requête : {query}[/bold cyan]\n")

    query_emb = model.encode(f"query: {query}", normalize_embeddings=True)
    hits = client.query_points(COLLECTION, query=query_emb.tolist(), limit=10, with_payload=True).points

    # 4. Top-3 children sans déduplication
    console.print("[bold]Top-3 children (vue brute)[/bold]")
    for i, hit in enumerate(hits[:3], start=1):
        console.print(Panel(
            hit.payload["child_content"][:150] + "...",
            title=f"Child #{i} (score={hit.score:.3f}, parent={hit.payload['parent_id']})",
            border_style="orange3",
        ))

    # 5. Déduplication par parent → on retourne les sections complètes au LLM
    # Si plusieurs enfants matchent le même parent, on ne garde que le parent une fois
    seen_parents: set[int] = set()
    unique_parents: list[tuple[int, str, float]] = []
    for hit in hits:
        pid = hit.payload["parent_id"]
        if pid not in seen_parents:
            seen_parents.add(pid)
            unique_parents.append((pid, hit.payload["parent_content"], hit.score))
        if len(unique_parents) >= 2:
            break

    console.print("\n[bold orange3]Top-2 parents uniques (ce qui sera passé au LLM)[/bold orange3]")
    for pid, content, score in unique_parents:
        console.print(Panel(
            content,
            title=f"Parent #{pid} (meilleur score child={score:.3f})",
            border_style="green",
        ))

    console.print("\n[bold]Pourquoi c'est mieux :[/bold]")
    console.print("• Le retrieval est précis car on cherche sur des chunks courts (200 tok)")
    console.print("• Mais le LLM voit du contexte large (600+ tok) → meilleures réponses")
    console.print("• Si plusieurs children matchent le même parent, on déduplique automatiquement")
    console.print("• Pattern recommandé pour la documentation technique structurée\n")


if __name__ == "__main__":
    main()
