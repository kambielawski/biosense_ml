"""Metric tracking utilities for training and validation."""

from collections import defaultdict


class MetricTracker:
    """Tracks running averages of metrics within an epoch.

    Usage:
        tracker = MetricTracker()
        for batch in dataloader:
            loss = compute_loss(...)
            tracker.update("loss", loss.item(), n=batch_size)
        print(tracker.all_averages())
    """

    def __init__(self) -> None:
        self._metrics: dict[str, dict[str, float]] = defaultdict(
            lambda: {"sum": 0.0, "count": 0}
        )

    def update(self, name: str, value: float, n: int = 1) -> None:
        """Record a metric value.

        Args:
            name: Metric name.
            value: Metric value (will be weighted by n).
            n: Number of samples this value represents.
        """
        self._metrics[name]["sum"] += value * n
        self._metrics[name]["count"] += n

    def average(self, name: str) -> float:
        """Get the running average of a metric.

        Args:
            name: Metric name.

        Returns:
            The weighted average.
        """
        m = self._metrics[name]
        if m["count"] == 0:
            return 0.0
        return m["sum"] / m["count"]

    def all_averages(self) -> dict[str, float]:
        """Get running averages for all tracked metrics."""
        return {name: self.average(name) for name in self._metrics}

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._metrics.clear()


def compute_accuracy(logits, targets) -> float:
    """Compute top-1 classification accuracy.

    Args:
        logits: Model output logits, shape (B, num_classes).
        targets: Ground truth labels, shape (B,).

    Returns:
        Accuracy as a float between 0 and 1.
    """
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()
