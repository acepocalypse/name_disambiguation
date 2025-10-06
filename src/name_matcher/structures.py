"""Basic data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DisjointSet:
    """Union-find structure for clustering."""

    size: int

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError("size must be non-negative")
        self.parent = list(range(self.size))

    def find(self, index: int) -> int:
        parent = self.parent[index]
        if parent != index:
            parent = self.find(parent)
            self.parent[index] = parent
        return parent

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        self.parent[root_left] = root_right
