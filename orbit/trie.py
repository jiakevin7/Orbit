from dataclasses import dataclass, field


@dataclass
class TrieNode:
    count: int = 0
    terminal_count: int = 0
    children: dict[int, "TrieNode"] = field(default_factory=dict)


class PrefixTrie:
    def __init__(self):
        # Counts let removal prune shared paths safely when cache entries evict.
        self.root = TrieNode()
        self.size = 0

    def insert(self, tokens):
        node = self.root
        node.count += 1
        for token in tokens:
            node = node.children.setdefault(token, TrieNode())
            node.count += 1
        node.terminal_count += 1
        self.size += 1

    def remove(self, tokens):
        nodes = [self.root]
        node = self.root
        for token in tokens:
            child = node.children.get(token)
            if child is None:
                raise KeyError("prefix is not present in the trie")
            node = child
            nodes.append(node)
        if node.terminal_count == 0:
            raise KeyError("prefix is not present in the trie")
        node.terminal_count -= 1
        for visited in nodes:
            visited.count -= 1
        for index in range(len(tokens), 0, -1):
            parent = nodes[index - 1]
            child = nodes[index]
            token = tokens[index - 1]
            if child.count == 0:
                del parent.children[token]
        self.size -= 1

    def longest_prefix(self, tokens):
        node = self.root
        depth = 0
        for token in tokens:
            child = node.children.get(token)
            if child is None or child.count == 0:
                break
            node = child
            depth += 1
        return depth

    def contains(self, tokens):
        node = self.root
        for token in tokens:
            node = node.children.get(token)
            if node is None or node.count == 0:
                return False
        return node.terminal_count > 0
