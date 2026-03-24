"""data — Dataset loading and federated data partitioning utilities."""

from .datasets import load_dataset, get_num_classes
from .partition import partition_data, get_client_stats

__all__ = ["load_dataset", "get_num_classes", "partition_data", "get_client_stats"]
