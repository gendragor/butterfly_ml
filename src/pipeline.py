"""Reproducible butterfly classification pipeline."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import joblib


def load_config(path: Path):
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def image_vector(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size))
        return np.asarray(im, dtype=np.float32).reshape(-1) / 255.0


def augmented_image(path: Path, seed: int, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        rng = random.Random(seed)
        if rng.random() < 0.5:
            im = ImageOps.mirror(im)
        im = im.rotate(rng.uniform(-15, 15), resample=Image.Resampling.BILINEAR)
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.15))
        im = im.resize((size, size))
        return np.asarray(im, dtype=np.float32).reshape(-1) / 255.0


def collect(root: Path, split: str, size: int, limit: int | None, augment: bool, copies: int, seed: int):
    split_root = root / split
    classes = sorted(p.name for p in split_root.iterdir() if p.is_dir())
    class_to_id = {name: i for i, name in enumerate(classes)}
    xs, ys = [], []
    for cls in classes:
        files = sorted((split_root / cls).glob("*.jpg"))
        if limit:
            files = files[:limit]
        for idx, path in enumerate(files):
            xs.append(image_vector(path, size)); ys.append(class_to_id[cls])
            if augment and split == "train":
                for copy in range(copies):
                    xs.append(augmented_image(path, seed + idx * 101 + copy, size)); ys.append(class_to_id[cls])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys), classes


def make_models(cfg):
    m = cfg["models"]
    return {
        "logistic_regression": LogisticRegression(C=m["logistic_regression"]["C"], max_iter=m["logistic_regression"]["max_iter"], solver="lbfgs", n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=m["random_forest"]["n_estimators"], max_depth=m["random_forest"]["max_depth"], n_jobs=m["random_forest"]["n_jobs"], random_state=cfg["project"]["seed"]),
        "svm": SVC(C=m["svm"]["C"], kernel=m["svm"]["kernel"], probability=m["svm"]["probability"]),
    }


def run(config_path: Path, output_root: Path):
    cfg = load_config(config_path); p = cfg["project"]; random.seed(p["seed"]); np.random.seed(p["seed"])
    dataset = Path(p["dataset_root"]); size = int(p["image_size"])
    x_train, y_train, classes = collect(dataset, "train", size, p.get("max_train_per_class"), p["use_augmentation"], p["augmented_copies_per_image"], p["seed"])
    x_valid, y_valid, _ = collect(dataset, "valid", size, None, False, 0, p["seed"])
    x_test, y_test, _ = collect(dataset, "test", size, None, False, 0, p["seed"])
    run_name = "augmented" if p["use_augmentation"] else "baseline"
    model_dir = output_root / "models" / run_name; report_dir = output_root / "reports" / run_name
    model_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, model in make_models(cfg).items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        results.append({"model": name, "accuracy": float(accuracy_score(y_test, pred)), "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)), "train_samples": int(len(y_train)), "test_samples": int(len(y_test))})
        joblib.dump({"model": model, "classes": classes, "image_size": size}, model_dir / f"{name}.joblib")
    (report_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run": run_name, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, required=True); ap.add_argument("--output-root", type=Path, default=Path(".")); args = ap.parse_args(); run(args.config, args.output_root)
