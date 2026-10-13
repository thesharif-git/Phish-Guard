"""
PhishGuard Vector RAG Engine (ChromaDB + all-MiniLM-L6-v2)
Semantic campaign memory database for zero-day phishing detection.
"""

import os
import logging
from threading import Lock
from typing import Optional, Dict, Any, List

logger = logging.getLogger("phishguard")

from core.config import settings

_CHROMADB_CLIENT = None
_COLLECTION = None
_EMBEDDING_MODEL = None
_INITIALIZED = False
_EMBEDDING_LOAD_LOCK = Lock()
_COLLECTION_LOAD_LOCK = Lock()


# Initial Zero-Day Campaign Seed Dataset (Cross-Lingual Vectors)
SEED_CAMPAIGNS = [
    {
        "id": "camp-m365-pass-expiry",
        "title": "Microsoft 365 Password Expiry Lure",
        "text": "Your Microsoft 365 account password will expire today. Please click the link to keep your current password and avoid losing mailbox access.",
        "category": "Credential Harvesting",
        "risk_level": "critical"
    },
    {
        "id": "camp-paypal-suspension",
        "title": "PayPal Account Suspension Threat",
        "text": "URGENT NOTICE: Your PayPal account has been temporarily restricted due to unauthorized login attempts. Confirm your identity immediately to restore access.",
        "category": "Financial Scam",
        "risk_level": "critical"
    },
    {
        "id": "camp-docusign-invoice",
        "title": "DocuSign Secure Document Signature Lure",
        "text": "You have received an urgent secure document from Accounts Payable via DocuSign. Click to review and sign the attached purchase invoice before expiration.",
        "category": "Document Phishing",
        "risk_level": "medium"
    },
    {
        "id": "camp-executive-wire-fraud",
        "title": "Executive Wire Transfer & CEO Fraud",
        "text": "I am in an urgent meeting and cannot take calls. Please process an immediate bank wire transfer of $18,500 to our vendor using the updated bank details.",
        "category": "BEC Wire Fraud",
        "risk_level": "critical"
    },
    {
        "id": "camp-bank-security-alert",
        "title": "Banking Security & Account Restriction Notice",
        "text": "Security Alert: Unusual transaction detected on your online banking account. Please log in immediately to verify your identity and prevent account block.",
        "category": "Banking Phishing",
        "risk_level": "critical"
    },
    {
        "id": "camp-it-helpdesk-reset",
        "title": "IT Helpdesk Credential Update Fraud",
        "text": "IT Department Notification: Mandatory corporate email password update required for all staff. Click here to confirm your current login details.",
        "category": "Corporate Spoofing",
        "risk_level": "medium"
    },
    # Multilingual Zero-Day Campaign Variants
    {
        "id": "camp-m365-expiry-es",
        "title": "Microsoft 365 Password Expiry (Spanish)",
        "text": "Aviso Urgente: Su cuenta de Microsoft 365 caducará hoy. Haga clic aquí para renovar su contraseña y evitar el bloqueo de su buzón.",
        "category": "Credential Harvesting",
        "risk_level": "critical"
    },
    {
        "id": "camp-paypal-suspension-fr",
        "title": "PayPal Account Suspension (French)",
        "text": "ALERTE DE SÉCURITÉ: Votre compte PayPal a été restreint suite à des tentatives de connexion non autorisées. Confirmez votre identité immédiatement.",
        "category": "Financial Scam",
        "risk_level": "critical"
    },
    {
        "id": "camp-bank-blocked-de",
        "title": "Bank Account Suspension (German)",
        "text": "Dringende Benachrichtigung: Ihr Bankkonto wurde vorübergehend gesperrt. Bitte verifizieren Sie Ihre Daten um die Sperrung aufzuheben.",
        "category": "Banking Phishing",
        "risk_level": "critical"
    },
    {
        "id": "camp-security-alert-ar",
        "title": "Account Security Alert (Arabic)",
        "text": "تنبيه أمني عاجل: تم تعليق حسابك مؤقتاً بسبب محاولات دخول غير مصرح بها. يرجى تأكيد هويتك فوراً لتجنب إغلاق الحساب.",
        "category": "Account Takeover",
        "risk_level": "critical"
    },
    {
        "id": "camp-banking-alert-ur",
        "title": "Banking Security Alert (Urdu)",
        "text": "فوری سیکیورٹی الرٹ: آپ کا بینک اکاؤنٹ عارضی طور پر معطل کر دیا گیا ہے۔ تصدیق کے لیے فوری طور پر اپنے اکاؤنٹ میں لاگ ان کریں۔",
        "category": "Banking Phishing",
        "risk_level": "critical"
    }
]


def _get_embedding_model():
    """
    Lazy load multilingual sentence-transformers model.
    Prioritizes 'paraphrase-multilingual-MiniLM-L12-v2' (Multilingual, 50+ languages)
    with seamless fallback to 'all-MiniLM-L6-v2'.
    """
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL if _EMBEDDING_MODEL else None

    with _EMBEDDING_LOAD_LOCK:
        if _EMBEDDING_MODEL is not None:
            return _EMBEDDING_MODEL if _EMBEDDING_MODEL else None
        return _load_embedding_model()


def _load_embedding_model():
    """Initialize while holding the embedding lock, including fallback state."""
    global _EMBEDDING_MODEL

    preferred_model = getattr(settings, "VECTOR_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    fallback_model = getattr(settings, "FALLBACK_VECTOR_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    try:
        from sentence_transformers import SentenceTransformer
        # Check local cache first to avoid hanging on network connections
        try:
            _EMBEDDING_MODEL = SentenceTransformer(preferred_model, local_files_only=True)
            logger.info(f"[VectorStore] Loaded primary multilingual SentenceTransformer from local cache: '{preferred_model}'")
        except Exception:
            try:
                _EMBEDDING_MODEL = SentenceTransformer(fallback_model, local_files_only=True)
                logger.info(f"[VectorStore] Loaded fallback SentenceTransformer from local cache: '{fallback_model}'")
            except Exception:
                logger.warning("[VectorStore] No cached embedding weights; campaign similarity unavailable")
                _EMBEDDING_MODEL = False
    except Exception as e:
        logger.error(f"[VectorStore] Failed to load SentenceTransformer: {e}")
        _EMBEDDING_MODEL = False

    return _EMBEDDING_MODEL if _EMBEDDING_MODEL else None


def _get_chroma_collection():
    """Lazy initialize ChromaDB persistent client and collection."""
    # The collection is assigned before seeding, so it is usable on the fast
    # path only after the entire initialization has finished.
    if _INITIALIZED and _COLLECTION is not None:
        return _COLLECTION

    with _COLLECTION_LOAD_LOCK:
        if _INITIALIZED and _COLLECTION is not None:
            return _COLLECTION
        return _initialize_chroma_collection()


def _initialize_chroma_collection():
    """Initialize/seed under the collection lock; embedding has its own lock."""
    global _CHROMADB_CLIENT, _COLLECTION, _INITIALIZED

    try:
        import chromadb
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chroma_dir = os.path.join(project_root, "data", "chroma_db")
        os.makedirs(chroma_dir, exist_ok=True)

        _CHROMADB_CLIENT = chromadb.PersistentClient(path=chroma_dir)
        _COLLECTION = _CHROMADB_CLIENT.get_or_create_collection(
            name="phishguard_campaigns",
            metadata={"hnsw:space": "cosine"}  # Cosine similarity (1.0 = identical, 0.0 = orthogonal)
        )
        logger.info(f"[VectorStore] ChromaDB initialized at {chroma_dir} (count: {_COLLECTION.count()})")

        # Seed initial zero-day campaign vectors if database is empty
        if _COLLECTION.count() == 0:
            _seed_initial_campaigns()

        _INITIALIZED = True
        return _COLLECTION
    except Exception as e:
        # A count/initialization failure can occur after assigning the client
        # and collection. Never publish those incomplete objects on a retry.
        _CHROMADB_CLIENT = None
        _COLLECTION = None
        _INITIALIZED = False
        logger.error(f"[VectorStore] Failed to initialize ChromaDB: {e}")
        return None


def _seed_initial_campaigns():
    """Seeds default zero-day campaign embeddings into ChromaDB."""
    model = _get_embedding_model()
    collection = _COLLECTION
    if not model or collection is None:
        return

    try:
        ids = [item["id"] for item in SEED_CAMPAIGNS]
        documents = [item["text"] for item in SEED_CAMPAIGNS]
        metadatas = [
            {
                "title": item["title"],
                "category": item["category"],
                "risk_level": item["risk_level"]
            }
            for item in SEED_CAMPAIGNS
        ]

        embeddings = model.encode(documents).tolist()
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"[VectorStore] Seeded {len(ids)} initial campaign vectors into ChromaDB.")
    except Exception as e:
        logger.error(f"[VectorStore] Error seeding campaigns: {e}")


def search_similar_campaigns(email_text: str, top_k: int = 3, threshold: float = 0.65) -> Dict[str, Any]:
    """
    Queries ChromaDB vector collection for semantic similarity against past campaign vectors.
    Cosine distance d ranges from 0 (identical) to 2 (opposite).
    Similarity = max(0.0, 1.0 - d).
    """
    if not email_text or len(email_text.strip()) < 10:
        return {"has_match": False, "match_percentage": 0, "matches": []}

    try:
        model = _get_embedding_model()
        if not model:
            return {"available": False, "has_match": False, "match_percentage": 0, "matches": [],
                    "reason": "Embedding weights/dependency unavailable", "verified_campaign_evidence": False}
        collection = _get_chroma_collection()
        if collection is None or collection.count() == 0:
            return {"available": False, "has_match": False, "match_percentage": 0, "matches": [],
                    "reason": "Campaign collection unavailable or empty", "verified_campaign_evidence": False}

        # Generate 384-dimensional query vector
        query_embedding = model.encode([email_text]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        matches = []
        best_similarity = 0.0
        best_title = None

        if results and "distances" in results and results["distances"]:
            distances = results["distances"][0]
            metadatas = results["metadatas"][0] if "metadatas" in results else []
            documents = results["documents"][0] if "documents" in results else []

            for dist, meta, doc in zip(distances, metadatas, documents):
                # Cosine distance d -> Cosine similarity = 1 - d
                sim = max(0.0, 1.0 - float(dist))
                sim_pct = round(sim * 100, 1)

                if sim >= best_similarity:
                    best_similarity = sim
                    best_title = meta.get("title", "Zero-Day Campaign Variant")

                if sim >= threshold:
                    matches.append({
                        "title": meta.get("title", "Zero-Day Campaign Variant"),
                        "category": meta.get("category", "Phishing Campaign"),
                        "risk_level": meta.get("risk_level", "critical"),
                        "similarity_score": round(sim, 4),
                        "similarity_percentage": sim_pct,
                        "snippet": doc[:150] + "..." if len(doc) > 150 else doc
                    })

        has_match = len(matches) > 0 or best_similarity >= threshold
        top_pct = round(best_similarity * 100, 1)

        return {
            "has_match": has_match,
            "available": True, "verified_campaign_evidence": False,
            "best_similarity": round(best_similarity, 4),
            "match_percentage": top_pct,
            "matched_title": best_title if has_match else None,
            "matches": matches
        }
    except Exception as e:
        logger.error(f"[VectorStore] Search error: {e}")
        return {"available": False, "has_match": False, "match_percentage": 0, "matches": [],
                "error": str(e), "verified_campaign_evidence": False}


def index_phishing_email(scan_id: str, text: str, title: str = "Detected Phishing Scan", category: str = "Recent Scan") -> bool:
    """Indexes a newly confirmed phishing scan vector into ChromaDB memory."""
    if not scan_id or not text or len(text.strip()) < 15:
        return False

    try:
        model = _get_embedding_model()
        collection = _get_chroma_collection()
        if not model or collection is None:
            return False

        doc_id = f"scan-{scan_id}"
        embedding = model.encode([text]).tolist()

        collection.upsert(
            ids=[doc_id],
            embeddings=embedding,
            documents=[text[:1000]],
            metadatas=[{
                "title": title,
                "category": category,
                "risk_level": "critical"
            }]
        )
        logger.info(f"[VectorStore] Indexed scan '{scan_id}' into ChromaDB persistent memory.")
        return True
    except Exception as e:
        logger.error(f"[VectorStore] Error indexing scan '{scan_id}': {e}")
        return False
