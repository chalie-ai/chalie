#!/usr/bin/env python3
"""
Verification script for STORY-05: GraphService and semantic query integration.

Tests:
1. GraphService initialization and methods
2. SemanticMemoryConfig type-safe access
3. act_dispatcher_service semantic_query action
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_graph_service():
    """Test GraphService methods."""
    print("\n=== Testing GraphService ===")

    from services.database_service import DatabaseService, get_merged_db_config
    from services.graph_service import GraphService

    try:
        # Initialize services
        db_config = get_merged_db_config()
        db_service = DatabaseService(db_config)
        graph_service = GraphService(db_service)

        # Test get_all_concepts
        print("Testing get_all_concepts()...")
        concepts = graph_service.get_all_concepts()
        print(f"✓ Retrieved {len(concepts)} concepts")

        if concepts:
            # Test get_concept
            print("\nTesting get_concept()...")
            concept_id = concepts[0]['id']
            concept = graph_service.get_concept(concept_id)
            if concept:
                print(f"✓ Retrieved concept: {concept['concept_name']} (type: {concept['concept_type']})")

            # Test get_relationships
            print("\nTesting get_relationships()...")
            relationships = graph_service.get_relationships(concept_id)
            print(f"✓ Retrieved {len(relationships)} relationships")

        db_service.close_pool()
        print("\n✓ GraphService tests passed")
        return True

    except Exception as e:
        print(f"\n✗ GraphService test failed: {e}")
        return False


def test_semantic_memory_config():
    """Test SemanticMemoryConfig type-safe access."""
    print("\n=== Testing SemanticMemoryConfig ===")

    from config.agents.semantic_memory_config import SemanticMemoryConfig

    try:
        config = SemanticMemoryConfig()

        # Test property access
        print(f"Embedding model: {config.embedding_model}")
        print(f"Embedding dimensions: {config.embedding_dimensions}")
        print(f"Min confidence threshold: {config.min_confidence_threshold}")
        print(f"Retrieval weights: {config.retrieval_weights}")

        # Test naming bridge
        assert config.retrieval_weights == config.inference_weights, "Naming bridge failed"

        print("\n✓ SemanticMemoryConfig tests passed")
        return True

    except Exception as e:
        print(f"\n✗ SemanticMemoryConfig test failed: {e}")
        return False


def test_semantic_query_action():
    """Test semantic_query action in ACT dispatcher."""
    print("\n=== Testing semantic_query Action ===")

    from services.act_dispatcher_service import ActDispatcherService

    try:
        dispatcher = ActDispatcherService(timeout=15.0)

        # Test semantic query action
        action = {
            'type': 'semantic_query',
            'query': 'test query',
            'limit': 3
        }

        print("Dispatching semantic_query action...")
        result = dispatcher.dispatch_action('test_topic', action)

        print(f"Status: {result['status']}")
        print(f"Result: {result['result'][:200]}...")
        print(f"Execution time: {result['execution_time']:.2f}s")

        if result['status'] == 'success':
            print("\n✓ Semantic query action executed successfully")
            return True
        else:
            print(f"\n✗ Semantic query action failed: {result['result']}")
            return False

    except Exception as e:
        print(f"\n✗ Semantic query action test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("STORY-05 Verification: GraphService and Semantic Query")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("GraphService", test_graph_service()))
    results.append(("SemanticMemoryConfig", test_semantic_memory_config()))
    results.append(("Semantic Query Action", test_semantic_query_action()))

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        all_passed = all_passed and passed

    print("=" * 60)

    if all_passed:
        print("\n✓ All verification tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed. Review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
