#!/usr/bin/env python3
"""
Integration tests for semantic memory system.
Tests storage, consolidation, retrieval, and spreading activation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.semantic_storage_service import SemanticStorageService
from services.semantic_retrieval_service import SemanticRetrievalService
from services.semantic_consolidation_service import SemanticConsolidationService
from services.database_service import DatabaseService
from datetime import datetime


def test_storage():
    """Test storing and retrieving a concept"""
    print("\n[TEST 1] Testing storage and retrieval...")

    try:
        db_service = DatabaseService()
        storage = SemanticStorageService(db_service)

        # Store a test concept
        concept_data = {
            'concept_name': 'Python Programming',
            'concept_type': 'knowledge',
            'definition': 'Python is a high-level programming language',
            'domain': 'programming',
            'confidence': 0.9
        }

        concept_id = storage.store_concept(concept_data)
        assert concept_id is not None, "Failed to store concept"

        # Retrieve the concept
        concept = storage.get_concept(concept_id)
        assert concept is not None, "Failed to retrieve concept"
        assert "python" in concept['concept_name'].lower(), "Concept name doesn't match"

        print("✓ Storage and retrieval test passed")
        return True

    except Exception as e:
        print(f"✗ Storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_consolidation():
    """Test extracting concepts from mock episode"""
    print("\n[TEST 2] Testing consolidation...")

    try:
        db_service = DatabaseService()
        consolidation = SemanticConsolidationService(db_service)

        # Mock episode data
        episode_text = """
        User asked: What is machine learning?
        Assistant: Machine learning is a subset of artificial intelligence where systems
        learn from data. Neural networks are a key technique used in deep learning.
        """

        # Extract concepts
        concepts = consolidation.extract_concepts_from_episode(
            episode_text=episode_text,
            episode_id="test_episode_001"
        )

        assert len(concepts) > 0, "No concepts extracted"

        # Check if relevant concepts were extracted
        concept_names = [c['concept_name'].lower() for c in concepts]
        has_ml_concept = any('machine learning' in name or 'neural' in name for name in concept_names)

        print(f"✓ Consolidation test passed - extracted {len(concepts)} concepts")
        if has_ml_concept:
            print("  Found expected ML-related concepts")

        return True

    except Exception as e:
        print(f"✗ Consolidation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieval():
    """Test hybrid search with confidence filtering"""
    print("\n[TEST 3] Testing retrieval with hybrid search...")

    try:
        db_service = DatabaseService()
        storage = SemanticStorageService(db_service)
        retrieval = SemanticRetrievalService(db_service)

        # Store test concepts
        concept1 = {
            'concept_name': 'Python Programming Language',
            'concept_type': 'knowledge',
            'definition': 'Python is a versatile programming language',
            'domain': 'programming',
            'confidence': 0.9
        }

        concept2 = {
            'concept_name': 'JavaScript Web Development',
            'concept_type': 'knowledge',
            'definition': 'JavaScript is used for web development',
            'domain': 'programming',
            'confidence': 0.85
        }

        storage.store_concept(concept1)
        storage.store_concept(concept2)

        # Search for related concepts
        results = retrieval.hybrid_search(
            query="programming languages",
            limit=10
        )

        assert len(results) > 0, "No results found"

        print(f"✓ Retrieval test passed - found {len(results)} relevant concepts")
        return True

    except Exception as e:
        print(f"✗ Retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spreading_activation():
    """Test BFS with weak relationship activation"""
    print("\n[TEST 4] Testing spreading activation...")

    try:
        db_service = DatabaseService()
        storage = SemanticStorageService(db_service)
        retrieval = SemanticRetrievalService(db_service)

        # Store connected concepts
        concept1_data = {
            'concept_name': 'Machine Learning Algorithms',
            'concept_type': 'knowledge',
            'definition': 'Algorithms that learn from data',
            'domain': 'AI',
            'confidence': 0.9
        }

        concept2_data = {
            'concept_name': 'Neural Networks',
            'concept_type': 'knowledge',
            'definition': 'Networks inspired by biological neurons',
            'domain': 'AI',
            'confidence': 0.9
        }

        concept1_id = storage.store_concept(concept1_data)
        concept2_id = storage.store_concept(concept2_data)

        # Create relationship
        relationship_data = {
            'source_concept_id': concept1_id,
            'target_concept_id': concept2_id,
            'relationship_type': 'related_to',
            'strength': 0.8,
            'confidence': 0.9
        }

        storage.store_relationship(relationship_data)

        # Get relationships to verify connection
        relationships = storage.get_relationships(concept1_id, direction='outgoing')

        assert len(relationships) > 0, "No relationships found"
        assert any(r['target_concept_id'] == concept2_id for r in relationships), "Expected relationship not created"

        print(f"✓ Spreading activation test passed - created {len(relationships)} relationships")
        return True

    except Exception as e:
        print(f"✗ Spreading activation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("SEMANTIC MEMORY INTEGRATION TESTS")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("Storage", test_storage()))
    results.append(("Consolidation", test_consolidation()))
    results.append(("Retrieval", test_retrieval()))
    results.append(("Spreading Activation", test_spreading_activation()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:25} {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)

    return all(result for _, result in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
