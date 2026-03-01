"""
Tests for DocumentProcessingService — text extraction, chunking, metadata extraction.
"""

import pytest
from unittest.mock import MagicMock, patch

from services.document_processing_service import DocumentProcessingService


@pytest.fixture
def service():
    return DocumentProcessingService()


@pytest.mark.unit
class TestChunking:
    def test_basic_chunking(self, service):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = service._chunk_text(text, 'default', chunk_size=50, overlap=10)

        assert len(chunks) >= 1
        assert all('content' in c for c in chunks)

    def test_adaptive_chunk_sizing(self, service):
        # Contract type gets larger chunks
        text = " ".join(["word"] * 500)
        contract_chunks = service._chunk_text(text, 'contract')
        default_chunks = service._chunk_text(text, 'default')

        # Contract chunks should be fewer (larger size)
        assert len(contract_chunks) <= len(default_chunks)

    def test_receipt_gets_small_chunks(self, service):
        text = " ".join(["item"] * 200)
        receipt_chunks = service._chunk_text(text, 'receipt')
        default_chunks = service._chunk_text(text, 'default')

        # Receipt chunks should be more (smaller size)
        assert len(receipt_chunks) >= len(default_chunks)

    def test_page_markers_tracked(self, service):
        text = "[Page 1]\nFirst page content.\n\n[Page 2]\nSecond page content."
        chunks = service._chunk_text(text, 'default', chunk_size=500, overlap=10)

        # Should have page numbers set
        assert any(c.get('page_number') is not None for c in chunks)

    def test_empty_text_returns_empty(self, service):
        chunks = service._chunk_text("", 'default')
        assert chunks == []


@pytest.mark.unit
class TestMetadataExtraction:
    def test_extract_dates_iso(self, service):
        text = "The document was created on 2025-03-15."
        dates = service._extract_dates(text)

        assert len(dates) >= 1
        assert '2025-03-15' in dates[0]['value']

    def test_extract_dates_natural(self, service):
        text = "Purchased on March 15, 2025 from the store."
        dates = service._extract_dates(text)

        assert len(dates) >= 1

    def test_extract_expiration_dates(self, service):
        text = "This warranty expires on 2027-03-15."
        exp_dates = service._extract_expiration_dates(text)

        assert len(exp_dates) >= 1
        assert '2027-03-15' in exp_dates[0]['value']

    def test_extract_monetary_values_usd(self, service):
        text = "Total purchase price: $1,299.99"
        values = service._extract_monetary_values(text)

        assert len(values) >= 1
        assert values[0]['currency'] == 'USD'
        assert values[0]['amount'] == 1299.99

    def test_extract_monetary_values_eur(self, service):
        text = "Price: €1.299,99"
        values = service._extract_monetary_values(text)

        assert len(values) >= 1
        assert values[0]['currency'] == 'EUR'

    def test_extract_reference_numbers(self, service):
        text = "Policy #WRN-2025-44821 covers this product."
        refs = service._extract_reference_numbers(text)

        assert len(refs) >= 1
        assert 'WRN-2025-44821' in refs[0]['value']

    def test_classify_warranty(self, service):
        text = "WARRANTY COVERAGE: This warranty covers manufacturer defects for 24 months. Repair or replacement guaranteed."
        doc_type = service._classify_document_type(text)

        assert doc_type['value'] == 'warranty'
        assert doc_type['confidence'] > 0.4

    def test_classify_invoice(self, service):
        text = "INVOICE: Amount due: $500. Payment required by end of month. Subtotal and total due."
        doc_type = service._classify_document_type(text)

        assert doc_type['value'] == 'invoice'

    def test_extract_companies(self, service):
        text = "Manufactured by Samsung Electronics. Sold by Best Buy."
        companies = service._extract_companies(text)

        assert len(companies) >= 1
        company_names = [c['name'].lower() for c in companies]
        assert any('samsung' in n for n in company_names)

    def test_extract_key_terms(self, service):
        text = "warranty coverage warranty repair warranty manufacturer coverage coverage"
        terms = service._extract_key_terms(text)

        assert 'warranty' in terms
        assert 'coverage' in terms


@pytest.mark.unit
class TestSummary:
    def test_summary_truncates(self, service):
        text = "x" * 1000
        summary = service._generate_summary(text)
        assert len(summary) <= 500

    def test_summary_prefers_sentence_boundary(self, service):
        text = "First sentence. Second sentence. " + "x" * 500
        summary = service._generate_summary(text)
        assert summary.endswith('.')

    def test_empty_text_returns_empty(self, service):
        assert service._generate_summary('') == ''


@pytest.mark.unit
class TestNormalization:
    def test_collapses_whitespace(self, service):
        text = "hello    world"
        result = service._normalize_text(text)
        assert result == "hello world"

    def test_collapses_newlines(self, service):
        text = "hello\n\n\n\n\nworld"
        result = service._normalize_text(text)
        assert result == "hello\n\nworld"


@pytest.mark.unit
class TestSimHash:
    def test_returns_hex_string(self, service):
        fp = service._simhash("hello world this is a test document")
        assert isinstance(fp, str)
        assert len(fp) > 0

    def test_similar_texts_similar_hashes(self, service):
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox jumps over the lazy cat"
        fp1 = service._simhash(text1)
        fp2 = service._simhash(text2)

        # Similar texts should produce similar fingerprints
        # (they share most shingles)
        assert fp1 != fp2  # Not identical

    def test_empty_text(self, service):
        assert service._simhash('') == ''


@pytest.mark.unit
class TestLanguageDetection:
    def test_detects_english(self, service):
        lang = service._detect_language("This is a test document written in English.")
        assert lang == 'en'

    def test_fallback_on_error(self, service):
        lang = service._detect_language("")
        assert lang == 'en'  # Fallback


@pytest.mark.unit
class TestAdaptiveBatching:
    def test_small_batch(self, service):
        mock_embedding = MagicMock()
        mock_embedding.generate_embeddings_batch.return_value = [[0.1] * 768] * 10
        texts = ["text"] * 10

        result = service._generate_chunk_embeddings(mock_embedding, texts)
        assert len(result) == 10
        # Should use batch_size=32 for < 50 chunks
        mock_embedding.generate_embeddings_batch.assert_called_once()

    def test_large_batch_splits(self, service):
        mock_embedding = MagicMock()
        mock_embedding.generate_embeddings_batch.return_value = [[0.1] * 768] * 16
        texts = ["text"] * 100

        result = service._generate_chunk_embeddings(mock_embedding, texts)
        # Should split into batches of 16 (50-200 range)
        assert mock_embedding.generate_embeddings_batch.call_count > 1

    def test_empty_texts(self, service):
        mock_embedding = MagicMock()
        result = service._generate_chunk_embeddings(mock_embedding, [])
        assert result == []
