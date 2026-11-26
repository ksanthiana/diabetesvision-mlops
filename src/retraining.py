
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


ROOT = Path(__file__).resolve().parents[1]

UPLOAD_DIR = ROOT / "data" / "uploaded"
MODEL_PATH = ROOT / "models" / "diabetesvision_mobilenetv2_final.keras"
OUTPUT_PATH = ROOT / "models" / "diabetesvision_mobilenetv2_retrained.keras"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5


def main():
    manifest = UPLOAD_DIR / "labels.csv"

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Original model not found: {MODEL_PATH}")

    if not manifest.exists():
        raise FileNotFoundError(
            "Add data/uploaded/labels.csv with columns: filename,diagnosis"
        )

    df = pd.read_csv(manifest)

    if not {"filename", "diagnosis"}.issubset(df.columns):
        raise ValueError("CSV must contain filename and diagnosis columns.")

    df["path"] = df["filename"].apply(lambda name: UPLOAD_DIR / name)
    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")

    if df["diagnosis"].isna().any() or not df["diagnosis"].isin(range(5)).all():
        raise ValueError("Diagnosis labels must be integers from 0 to 4.")

    if not df["path"].apply(Path.exists).all():
        raise FileNotFoundError("Some image files listed in labels.csv are missing.")

    if len(df) < 10 or df["diagnosis"].nunique() < 2:
        raise ValueError("Upload at least 10 labelled images from at least 2 grades.")

    counts = df["diagnosis"].value_counts()
    if counts.min() < 2:
        raise ValueError("Each represented grade needs at least 2 images.")

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["diagnosis"]
    )

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, tf.cast(label, tf.int32)

    def make_dataset(frame, shuffle=False):
        paths = frame["path"].astype(str).values
        labels = frame["diagnosis"].astype("int32").values

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            dataset = dataset.shuffle(len(frame), seed=42)

        return (
            dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

    train_ds = make_dataset(train_df, shuffle=True)
    val_ds = make_dataset(val_df)

    model = tf.keras.models.load_model(MODEL_PATH)

    classes = np.sort(train_df["diagnosis"].unique())
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["diagnosis"].values
    )
    class_weights = {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    model.save(OUTPUT_PATH)

    report = {
        "training_images": len(train_df),
        "validation_images": len(val_df),
        "epochs_completed": len(history.history["loss"]),
        "final_training_accuracy": float(history.history["accuracy"][-1]),
        "final_validation_accuracy": float(history.history["val_accuracy"][-1]),
        "model_path": str(OUTPUT_PATH)
    }

    report_path = ROOT / "models" / "retraining_report.json"
    report_path.write_text(json.dumps(report, indent=4))

    print("Retraining completed successfully!")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()
