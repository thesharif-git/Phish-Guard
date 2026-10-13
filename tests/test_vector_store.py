import os
import sys
import pytest

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vector_store import (
    search_similar_campaigns,
    index_phishing_email,
    _get_embedding_model,
    _get_chroma_collection
)


def test_embedding_model_loading():
    """Verify SentenceTransformer all-MiniLM-L6-v2 loads properly."""
    model = _get_embedding_model()
    assert model is not None
    embeddings = model.encode(["Test phishing email string"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384  # 384-dimensional vector


def test_chroma_collection_initialization():
    """Verify ChromaDB collection initializes and seeds initial campaign vectors."""
    collection = _get_chroma_collection()
    assert collection is not None
    assert collection.count() >= 6  # Seeded initial campaigns


def test_vector_similarity_search():
    """Verify semantic similarity search against zero-day phishing campaigns."""
    query_text = "Your Microsoft 365 password will expire today. Please click here to keep your current password."
    res = search_similar_campaigns(query_text)

    assert "has_match" in res
    assert res["has_match"] is True
    assert res["match_percentage"] > 70.0
    assert res["matched_title"] is not None
    assert "Microsoft" in res["matched_title"]


def test_index_new_phishing_scan():
    """Verify indexing a new confirmed scan into ChromaDB persistent memory."""
    scan_id = "test-scan-999"
    phish_text = "URGENT TAX AUDIT NOTICE: Download your IRS tax audit statement at http://irs-fraud-portal.xyz immediately."
    title = "IRS Tax Audit Phishing Scam"

    success = index_phishing_email(scan_id, phish_text, title=title)
    assert success is True

    # Query the newly indexed scan text
    search_res = search_similar_campaigns("URGENT TAX AUDIT NOTICE: Download tax audit statement at irs portal")
    assert search_res["has_match"] is True
    assert search_res["match_percentage"] >= 65.0
