"""Core pipeline for the Name Matcher library."""

from __future__ import annotations

import concurrent.futures
import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from tqdm import tqdm

    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _TQDM_AVAILABLE = False

from .canonical import generate_clean_canonical
from .comparison import clusters_conflict, is_plausible_expansion
from .llm import LLMClient, LLMConfig
from .normalization import get_block_key, normalize
from .structures import DisjointSet


@dataclass
class NameMatcherStats:
    """Summary metrics for a Name Matcher run."""

    total_names: int
    candidate_pairs: int
    merges_by_reason: Dict[str, int]
    rejections_by_reason: Dict[str, int]
    outliers_ejected: int
    cluster_count: int
    runtime_seconds: float


@dataclass
class NameMatcherResult:
    """Result bundle returned by :class:NameMatcher."""

    dataframe: pd.DataFrame
    cluster_map: Dict[int, List[int]]
    stats: NameMatcherStats


@dataclass
class NameMatcherConfig:
    """Configuration parameters for :class:NameMatcher."""

    name_column: str = "name"
    auto_no_prob: float = 0.35
    tfidf_analyzer: str = "char"
    tfidf_ngram_range: Tuple[int, int] = (2, 4)
    use_tqdm: bool | None = None
    verbose: bool = True
    llm_token: str | None = None
    llm: LLMConfig = field(default_factory=LLMConfig)


class NameMatcher:
    """Cluster records that likely refer to the same person."""

    def __init__(self, config: NameMatcherConfig | None = None, llm_client: LLMClient | None = None) -> None:
        self.config = config or NameMatcherConfig()
        if self.config.llm_token is not None:
            self.config.llm.token = self.config.llm_token
            self.config.llm.enabled = True
        self.llm_client = llm_client or LLMClient(self.config.llm)

    def cluster(
        self,
        dataframe: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> NameMatcherResult:
        """Perform clustering, optionally save results, and return the enriched dataframe."""

        if self.config.name_column not in dataframe.columns:
            raise KeyError(f"Column '{self.config.name_column}' not found in dataframe")

        verbose = self.config.verbose
        overall_start_time = time.time()
        if verbose:
            print("--- Name Matcher Process Started (v6.8.8 Improved Canonical Name Selection) ---")
            print("\n1. Loading and preparing data...")

        t0 = time.time()
        df = dataframe.copy()
        df[self.config.name_column] = df[self.config.name_column].fillna("").astype(str)
        df["name_norm"] = df[self.config.name_column].map(normalize)
        originals = df[self.config.name_column].tolist()
        names = df["name_norm"].tolist()
        if verbose:
            print(f"   Loaded {len(df)} names. Done in {time.time() - t0:.2f}s")

        t0 = time.time()
        if verbose:
            print("2. Vectorizing names for similarity scoring...")
        vectorizer = TfidfVectorizer(
            analyzer=self.config.tfidf_analyzer,
            ngram_range=self.config.tfidf_ngram_range,
        )
        matrix = vectorizer.fit_transform(names)
        if verbose:
            print(f"   Done in {time.time() - t0:.2f}s")

        t0 = time.time()
        if verbose:
            print("3. Generating candidate pairs via Blocking...")
        candidate_pairs_set = self._build_candidate_pairs(names)
        candidate_pairs_list = list(candidate_pairs_set)
        if verbose:
            print(f"   Generated {len(candidate_pairs_set)} candidate pairs from blocking.")
            print(f"   Done in {time.time() - t0:.2f}s")

        t0_filter = time.time()
        if verbose:
            print("4. Applying Stricter Cluster-Centric Filtering...")
        clusters = DisjointSet(len(names))
        temp_cluster_map: Dict[int, List[int]] = {i: [i] for i in range(len(names))}
        stats_counter: defaultdict[str, int] = defaultdict(int)
        pairs_for_llm: List[Dict[str, object]] = []

        iterator: Iterable[Tuple[int, int]] = candidate_pairs_list
        if candidate_pairs_list and self._use_tqdm:
            iterator = tqdm(candidate_pairs_list, desc="   Filtering Pairs", unit="pair")

        for left, right in iterator:
            if clusters.find(left) == clusters.find(right):
                continue
            if self._clusters_conflict(left, right, clusters, temp_cluster_map, names):
                stats_counter["rejected_transitive_conflict"] += 1
                continue
            if is_plausible_expansion(names[left], names[right]):
                self._union_clusters(clusters, temp_cluster_map, left, right)
                stats_counter["merged_expansion"] += 1
                continue
            pairs_for_llm.append(
                {
                    "i": left,
                    "j": right,
                    "key": tuple(sorted((originals[left], originals[right]))),
                }
            )

        if verbose:
            print(f"   Filter Stats: {dict(stats_counter)}")
            print(f"   Done in {time.time() - t0_filter:.2f}s")

        t0_llm = time.time()
        if verbose:
            print("5. Reviewing ambiguous pairs with LLM...")
        self._run_llm_review(pairs_for_llm, stats_counter, clusters, temp_cluster_map, names)
        if verbose:
            print(f"   Done in {time.time() - t0_llm:.2f}s")

        t0 = time.time()
        if verbose:
            print("6. Finalizing cluster data structures...")
        final_clusters_map = self._build_final_cluster_map(clusters, len(names))
        if verbose:
            print(f"   Done in {time.time() - t0:.2f}s")

        t0 = time.time()
        if verbose:
            print("7. Refining clusters by ejecting outliers...")
        outliers_found = self._eject_outliers(final_clusters_map, clusters, matrix)
        stats_counter["outliers_ejected"] = outliers_found
        if verbose:
            print(f"   Found and ejected {outliers_found} outliers from clusters.")
            print(f"   Done in {time.time() - t0:.2f}s")

        final_clusters_map = self._build_final_cluster_map(clusters, len(names))

        t0 = time.time()
        if verbose:
            print("8. Assigning canonical names and saving results...")
        df["cluster_id"] = df.index.map(lambda idx: clusters.find(idx))
        cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, root in enumerate(df["cluster_id"]):
            cluster_to_indices[root].append(idx)
        cluster_to_label = {
            root: generate_clean_canonical([originals[i] for i in indices])
            for root, indices in cluster_to_indices.items()
        }
        df["cluster_label"] = df["cluster_id"].map(cluster_to_label)

        if verbose:
            print("\n--- Results Summary ---")
            num_clusters = len([members for members in final_clusters_map.values() if members])
            print(f"   - Total names processed: {len(df)}")
            print(f"   - Unique entities (clusters) found: {num_clusters}")
            clusters_by_size = sorted(final_clusters_map.values(), key=len, reverse=True)
            print("\n   --- Sample of Largest Clusters Found ---")
            for idx, cluster_indices in enumerate(clusters_by_size[:10]):
                if len(cluster_indices) <= 1:
                    break
                canonical_name = df.loc[cluster_indices[0], "cluster_label"]
                print(f"   Cluster {idx + 1} (Size: {len(cluster_indices)}): '{canonical_name}'")
                for original_idx in cluster_indices[:5]:
                    print(f"     - {df.loc[original_idx, self.config.name_column]}")
                if len(cluster_indices) > 5:
                    print("     - ...")

        if output_path is not None:
            output_str = str(output_path)
            self._save_dataframe(df, output_str)
            if verbose:
                print(f"\n   Processing complete. Results saved to '{output_str}'")

        if verbose:
            print(f"   Done in {time.time() - t0:.2f}s")

        final_cluster_map = self._build_final_cluster_map(clusters, len(names))

        elapsed = time.time() - overall_start_time
        summary = NameMatcherStats(
            total_names=len(df),
            candidate_pairs=len(candidate_pairs_set),
            merges_by_reason={k: v for k, v in stats_counter.items() if k.startswith("merged")},
            rejections_by_reason={k: v for k, v in stats_counter.items() if k.startswith("rejected")},
            outliers_ejected=outliers_found,
            cluster_count=len([members for members in final_cluster_map.values() if members]),
            runtime_seconds=elapsed,
        )

        if verbose:
            print(f"\n--- Name Matcher Process Finished in {elapsed:.2f} seconds ---")

        return NameMatcherResult(dataframe=df, cluster_map=final_cluster_map, stats=summary)

    @property
    def _use_tqdm(self) -> bool:
        if self.config.use_tqdm is not None:
            return self.config.use_tqdm and _TQDM_AVAILABLE
        return _TQDM_AVAILABLE

    @staticmethod
    def _build_candidate_pairs(names: Sequence[str]) -> set[Tuple[int, int]]:
        blocks: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for index, name in enumerate(names):
            key = get_block_key(name)
            if key != ("", ""):
                blocks[key].append(index)

        pairs: set[Tuple[int, int]] = set()
        for indices in blocks.values():
            if len(indices) < 2:
                continue
            for left, right in itertools.combinations(indices, 2):
                pairs.add(tuple(sorted((left, right))))
        return pairs

    @staticmethod
    def _clusters_conflict(
        idx1: int,
        idx2: int,
        clusters: DisjointSet,
        cluster_map: Dict[int, List[int]],
        normalized_names: Sequence[str],
    ) -> bool:
        if clusters.find(idx1) == clusters.find(idx2):
            return False
        root_left = clusters.find(idx1)
        root_right = clusters.find(idx2)
        members_left = cluster_map[root_left]
        members_right = cluster_map[root_right]
        return clusters_conflict(members_left, members_right, normalized_names)

    @staticmethod
    def _union_clusters(
        clusters: DisjointSet,
        cluster_map: Dict[int, List[int]],
        left: int,
        right: int,
    ) -> None:
        root_left = clusters.find(left)
        root_right = clusters.find(right)
        clusters.union(left, right)
        new_root = clusters.find(left)
        if new_root == root_left:
            cluster_map[root_left].extend(cluster_map.pop(root_right, []))
        else:
            cluster_map[root_right].extend(cluster_map.pop(root_left, []))

    def _run_llm_review(
        self,
        pairs_for_llm: List[Dict[str, object]],
        stats_counter: defaultdict[str, int],
        clusters: DisjointSet,
        cluster_map: Dict[int, List[int]],
        normalized_names: Sequence[str],
    ) -> None:
        if not pairs_for_llm:
            return

        llm_cache: Dict[Tuple[str, str], bool | None] = {}
        unique_pairs_for_llm = [
            pair
            for pair in pairs_for_llm
            if pair["key"] not in llm_cache and not llm_cache.update({pair["key"]: None})
        ]

        if unique_pairs_for_llm:
            total_to_process = len(unique_pairs_for_llm)
            if self.config.verbose:
                print(
                    f"   Sending {total_to_process} unique pairs to LLM for review using "
                    f"{self.config.llm.concurrent_requests} parallel workers..."
                )
            unique_keys = [pair["key"] for pair in unique_pairs_for_llm]
            batch_size = max(1, self.config.llm.batch_size)
            batches = [unique_keys[i : i + batch_size] for i in range(0, len(unique_keys), batch_size)]

            if self.config.llm.concurrent_requests > 1 and batches:
                executor_cm = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.config.llm.concurrent_requests
                )
            else:
                executor_cm = concurrent.futures.ThreadPoolExecutor(max_workers=1)

            with executor_cm as executor:
                future_to_batch = {
                    executor.submit(self.llm_client.ask_batch, batch): batch for batch in batches
                }
                results_iterator = concurrent.futures.as_completed(future_to_batch)
                if batches and self._use_tqdm:
                    results_iterator = tqdm(results_iterator, total=len(batches), desc="   LLM Review", unit="batch")
                for future in results_iterator:
                    batch_keys = future_to_batch[future]
                    try:
                        decisions = future.result()
                        for key, decision in zip(batch_keys, decisions):
                            llm_cache[key] = decision
                    except Exception as exc:  # pragma: no cover - safeguard
                        print(f"\n   A batch generated an exception: {exc}. Defaulting to \"no\" for that batch.")
                        for key in batch_keys:
                            llm_cache[key] = False

        for pair_info in pairs_for_llm:
            key = pair_info["key"]  # type: ignore[index]
            if not llm_cache.get(key, False):
                continue
            left = pair_info["i"]  # type: ignore[index]
            right = pair_info["j"]  # type: ignore[index]
            if clusters.find(left) == clusters.find(right):
                continue
            if self._clusters_conflict(left, right, clusters, cluster_map, normalized_names):
                stats_counter["rejected_llm_post_conflict"] += 1
                continue
            self._union_clusters(clusters, cluster_map, left, right)
            stats_counter["merged_llm"] += 1

    @staticmethod
    def _build_final_cluster_map(clusters: DisjointSet, size: int) -> Dict[int, List[int]]:
    
        final_map: Dict[int, List[int]] = defaultdict(list)
        for index in range(size):
            root = clusters.find(index)
            final_map[root].append(index)
        return dict(final_map)

    def _eject_outliers(
        self,
        cluster_map: Dict[int, List[int]],
        clusters: DisjointSet,
        matrix,
    ) -> int:
        outliers = 0
        for root, indices in list(cluster_map.items()):
            if len(indices) < 2:
                continue
            centroid = np.asarray(matrix[indices, :].mean(axis=0))
            similarities = cosine_similarity(matrix[indices, :], centroid.reshape(1, -1)).flatten()
            to_remove = [index for index, score in zip(indices, similarities) if score < self.config.auto_no_prob]
            if not to_remove:
                continue
            for idx in to_remove:
                if idx not in cluster_map[root]:
                    continue
                cluster_map[root].remove(idx)
                clusters.parent[idx] = idx
                cluster_map[idx] = [idx]
                outliers += 1
        return outliers

    @staticmethod
    def _save_dataframe(dataframe: pd.DataFrame, output_path: str | Path) -> None:
        path = Path(output_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            dataframe.to_csv(path, index=False)
            return
        if suffix in {".xls", ".xlsx"}:
            dataframe.to_excel(path, index=False)
            return
        raise ValueError(f"Unsupported output file format: '{suffix}'")


__all__ = [
    "NameMatcher",
    "NameMatcherConfig",
    "NameMatcherResult",
    "NameMatcherStats",
]

