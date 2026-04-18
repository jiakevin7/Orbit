from __future__ import annotations

import unittest

from orbit.trie import PrefixTrie


class PrefixTrieTests(unittest.TestCase):
    def test_longest_prefix_tracks_shared_leading_tokens(self) -> None:
        trie = PrefixTrie()
        trie.insert((1, 2, 3, 4))
        trie.insert((1, 2, 9))

        self.assertEqual(trie.longest_prefix((1, 2, 3, 8)), 3)
        self.assertEqual(trie.longest_prefix((1, 2, 5)), 2)
        self.assertEqual(trie.longest_prefix((7, 8, 9)), 0)

    def test_contains_is_exact_not_prefix_only(self) -> None:
        trie = PrefixTrie()
        trie.insert((1, 2, 3))

        self.assertTrue(trie.contains((1, 2, 3)))
        self.assertFalse(trie.contains((1, 2)))

    def test_remove_prunes_dead_paths(self) -> None:
        trie = PrefixTrie()
        trie.insert((1, 2, 3))
        trie.insert((1, 2, 4))
        trie.remove((1, 2, 3))

        self.assertFalse(trie.contains((1, 2, 3)))
        self.assertEqual(trie.longest_prefix((1, 2, 4, 5)), 3)


if __name__ == "__main__":
    unittest.main()

