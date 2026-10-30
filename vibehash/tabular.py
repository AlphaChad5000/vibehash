"""Tables in, predictions out: what `vibehash train`, `job download` and `predict` use.

A CSV with a header row becomes a training dataset: every numeric column is an
input, and one column (the last, unless named) is the label to predict, as
numbers or as names. Inputs are standardised, because the network learns far
better from features on similar scales, and the scaling is saved with the model
so predictions apply the same one.

A downloaded model is one .npz holding the weights, the layer widths, the
scaling, and the input and class names, so `predict` needs nothing else.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import Dataset, ModelSpec, _forward


class TableError(ValueError):
    pass


@dataclass
class Table:
    features: list[str]
    label: str | None  # None when the file has no label column (data to predict on)
    x: np.ndarray
    y: list[str] | None  # labels as written in the file


def read_csv(path: str | Path, label: str | None = None, features: list[str] | None = None) -> Table:
    """Read a CSV. With `features` (from a trained model), take those columns by name and use `label` only if present."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if any(cell.strip() for cell in r)]
    if len(rows) < 2:
        raise TableError(f"{path} needs a header row and at least one data row")
    header = [h.strip() for h in rows[0]]
    if len(set(header)) != len(header):
        raise TableError("column names in the header must be unique")
    if features is None:
        label = label or header[-1]
        if label not in header:
            raise TableError(f"no column named {label!r}; the columns are: {', '.join(header)}")
        features = [h for h in header if h != label]
    else:
        missing = [h for h in features if h not in header]
        if missing:
            raise TableError(f"{path} is missing input columns the model needs: {', '.join(missing)}")
        label = label if label in header else None
    cols = [header.index(h) for h in features]
    x = np.empty((len(rows) - 1, len(cols)))
    for i, row in enumerate(rows[1:]):
        if len(row) != len(header):
            raise TableError(f"row {i + 2} has {len(row)} values but the header has {len(header)}")
        for j, c in enumerate(cols):
            try:
                x[i, j] = float(row[c])
            except ValueError:
                raise TableError(f"row {i + 2}, column {header[c]!r}: {row[c]!r} isn't a number. "
                                 "Inputs must be numbers; leave text columns out or turn them into numbers.") from None
    if not np.all(np.isfinite(x)):
        raise TableError("inputs must be finite numbers (no blanks, NaN or infinity)")
    y = [row[header.index(label)].strip() for row in rows[1:]] if label else None
    return Table(features, label, x, y)


def class_names(labels: list[str]) -> list[str]:
    """Classes in a stable order: numerically if every label is a number, else alphabetically."""
    unique = set(labels)
    try:
        return sorted(unique, key=float)
    except ValueError:
        return sorted(unique)


def to_dataset(table: Table, val_fraction: float = 0.2, seed: int = 0):
    """Shuffle, split off a validation set, standardise. Returns (Dataset, classes, mean, std)."""
    if table.y is None:
        raise TableError("training needs a label column")
    classes = class_names(table.y)
    if len(classes) < 2:
        raise TableError(f"the label column {table.label!r} has only one value; there's nothing to tell apart")
    if len(classes) > 1000:
        raise TableError(f"the label column {table.label!r} has {len(classes)} different values; at most 1000 classes")
    if not 1 <= len(table.features) <= 4096:
        raise TableError(f"between 1 and 4096 input columns are allowed, not {len(table.features)}")
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[v] for v in table.y], dtype=np.int64)
    n = len(y)
    if n < 10:
        raise TableError(f"{n} rows is too few to train and check a model; use at least 10")
    order = np.random.default_rng(seed).permutation(n)
    n_val = max(2, int(round(n * val_fraction)))
    val, train = order[:n_val], order[n_val:]
    mean = table.x[train].mean(axis=0)
    std = table.x[train].std(axis=0)
    std[std == 0] = 1.0  # a constant column carries no information; leave it at zero
    xs = (table.x - mean) / std
    return Dataset(xs[train], y[train], xs[val], y[val]), classes, mean, std


def write_predictions(src: str | Path, out: str | Path, predicted: list[str], confidence: np.ndarray) -> Path:
    """The input CSV with two columns added: what the model predicts for each row, and how sure it is."""
    with open(src, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if any(cell.strip() for cell in r)]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(rows[0] + ["predicted", "confidence"])
        for row, p, c in zip(rows[1:], predicted, confidence):
            w.writerow(row + [p, f"{c:.3f}"])
    return Path(out)


def save_dataset(data: Dataset, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x_train=data.x_train, y_train=data.y_train, x_val=data.x_val, y_val=data.y_val)
    return path


# --- trained models ------------------------------------------------------------------

@dataclass
class Model:
    spec: ModelSpec
    weights: np.ndarray
    features: list[str]
    classes: list[str]
    mean: np.ndarray
    std: np.ndarray
    info: dict

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities, one row per input row, columns in `classes` order."""
        _, probs = _forward(self.spec, self.weights, (np.asarray(x, dtype=np.float64) - self.mean) / self.std)
        return probs


def save_model(path: str | Path, spec: ModelSpec, weights: np.ndarray, features: list[str], classes: list[str],
               mean: np.ndarray, std: np.ndarray, info: dict) -> Path:
    path = Path(path)
    np.savez(path, weights=np.asarray(weights, dtype=np.float64), widths=np.array(spec.widths, dtype=np.int64),
             precision=np.array(spec.precision), features=np.array(features, dtype=str), classes=np.array(classes, dtype=str),
             mean=np.asarray(mean, dtype=np.float64), std=np.asarray(std, dtype=np.float64), info=np.array(json.dumps(info)))
    return path


def load_model(path: str | Path) -> Model:
    try:
        with np.load(path, allow_pickle=False) as f:  # a model file may come from anyone: never unpickle
            widths = [int(w) for w in f["widths"]]
            spec = ModelSpec(widths[0], tuple(widths[1:-1]), widths[-1], str(f["precision"]))
            weights = np.asarray(f["weights"], dtype=np.float64)
            model = Model(spec, weights, [str(s) for s in f["features"]], [str(s) for s in f["classes"]],
                          np.asarray(f["mean"], dtype=np.float64), np.asarray(f["std"], dtype=np.float64),
                          json.loads(str(f["info"])))
    except (OSError, KeyError, ValueError) as e:
        raise TableError(f"{path} isn't a Vibe Hash model file ({e})") from e
    if weights.shape != (spec.n_params,) or len(model.features) != spec.n_in or len(model.classes) != spec.n_out:
        raise TableError(f"{path} is damaged: its weights don't match its layer sizes")
    return model
