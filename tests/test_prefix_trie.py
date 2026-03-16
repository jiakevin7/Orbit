import pytest

from orbit.router.prefix_trie import PrefixTrie


class TestPrefixTrie:
    def test_empty_lookup(self):
        trie = PrefixTrie()
        assert trie.lookup(["hash1", "hash2"]) == {}

    def test_insert_and_lookup(self):
        trie = PrefixTrie()
        trie.insert(["h1", "h2", "h3"], "replica-1")

        matches = trie.lookup(["h1", "h2", "h3"])
        assert matches == {"replica-1": 3}

    def test_partial_match(self):
        trie = PrefixTrie()
        trie.insert(["h1", "h2", "h3"], "replica-1")

        # Query with longer prefix — only matches first 3
        matches = trie.lookup(["h1", "h2", "h3", "h4"])
        assert matches == {"replica-1": 3}

    def test_shorter_query(self):
        trie = PrefixTrie()
        trie.insert(["h1", "h2", "h3"], "replica-1")

        # Query is shorter than cached prefix
        matches = trie.lookup(["h1", "h2"])
        assert matches == {"replica-1": 2}

    def test_no_match(self):
        trie = PrefixTrie()
        trie.insert(["h1", "h2", "h3"], "replica-1")

        # Completely different prefix
        matches = trie.lookup(["x1", "x2"])
        assert matches == {}

    def test_multiple_replicas(self):
        trie = PrefixTrie()
        trie.insert(["h1", "h2", "h3"], "replica-1")
        trie.insert(["h1", "h2", "h4"], "replica-2")

        # Both match first 2 blocks
        matches = trie.lookup(["h1", "h2", "h5"])
        assert matches == {"replica-1": 2, "replica-2": 2}

    def test_different_match_depths(self):
        trie = PrefixTrie()
        trie.insert(["h1", "h2", "h3"], "replica-1")
        trie.insert(["h1", "h2", "h3", "h4", "h5"], "replica-2")

        # replica-2 has a longer match
        matches = trie.lookup(["h1", "h2", "h3", "h4", "h5"])
        assert matches["replica-1"] == 3
        assert matches["replica-2"] == 5

    def test_remove_replica(self):
        trie = PrefixTrie()
        trie.insert(["h1", "h2", "h3"], "replica-1")
        trie.insert(["h1", "h2", "h3"], "replica-2")

        trie.remove_replica(["h1", "h2", "h3"], "replica-1")

        matches = trie.lookup(["h1", "h2", "h3"])
        assert "replica-1" not in matches
        assert matches == {"replica-2": 3}

    def test_remove_replica_block(self):
        trie = PrefixTrie()
        trie.insert(["h1", "h2", "h3"], "replica-1")

        trie.remove_replica_block("h2", "replica-1")

        # After removing h2, lookup should still match h1 but stop there
        # because replica-1 is removed from h2 node
        matches = trie.lookup(["h1", "h2", "h3"])
        assert matches == {"replica-1": 1}

    def test_empty_insert(self):
        trie = PrefixTrie()
        trie.insert([], "replica-1")
        assert trie.lookup(["h1"]) == {}

    def test_get_all_replicas(self):
        trie = PrefixTrie()
        trie.insert(["h1"], "replica-1")
        trie.insert(["h2"], "replica-2")
        trie.insert(["h1", "h3"], "replica-3")

        replicas = trie.get_all_replicas()
        assert replicas == {"replica-1", "replica-2", "replica-3"}
