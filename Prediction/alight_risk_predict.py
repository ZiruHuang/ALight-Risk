#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALight-Risk command-line predictor.

Workflow
--------
1. Read an input FASTA file and verify unique sequence IDs.
2. Extract IMGT-numbered variable light-chain sequences with AbNumber.
3. Separate kappa and lambda sequences.
4. Calculate CKSAAP descriptors with iFeature.
5. Align descriptors to the fixed feature lists used during training.
6. Apply the fixed scaler and trained SVM model for each chain type.
7. Write sequence-level AL-risk probabilities and binary predictions.

This script does not rerun feature selection or model training.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
from abnumber import Chain


MODEL_FILENAMES = {
    "Kappa": {
        "template": "Kappa_trainset-1_CKSAAP_reduce.csv",
        "model": "kappa_1CKSAAP_SVM.pkl",
        "scaler": "kappa_1CKSAAP_scaler.pkl",
    },
    "Lambda": {
        "template": "Lambda_trainset-1_CKSAAP_reduce.csv",
        "model": "lambda_1CKSAAP_SVM.pkl",
        "scaler": "lambda_1CKSAAP_scaler.pkl",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict immunoglobulin light-chain amyloidogenic risk from a FASTA file "
            "using the fixed ALight-Risk kappa and lambda models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Input FASTA file containing one or more immunoglobulin light-chain sequences.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("alight_risk_result"),
        help="Directory for prediction results.",
    )
    parser.add_argument(
        "--prefix",
        default="ALight-Risk",
        help="Prefix used for output files.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing the fixed feature templates, trained SVM models, "
            "and fitted scalers."
        ),
    )
    parser.add_argument(
        "--ifeature-script",
        required=True,
        type=Path,
        help="Path to iFeature.py.",
    )
    parser.add_argument(
        "--ifeature-python",
        default=sys.executable,
        help="Python executable used to run iFeature.py.",
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=max(1, min(os.cpu_count() or 1, 8)),
        help="Number of worker processes used for AbNumber processing.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to assign the binary prediction label.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep chain-specific FASTA and iFeature output files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input FASTA file not found: {args.input}")
    if not args.ifeature_script.is_file():
        raise FileNotFoundError(f"iFeature script not found: {args.ifeature_script}")
    if not args.model_dir.is_dir():
        raise NotADirectoryError(f"Model directory not found: {args.model_dir}")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    missing = []
    for chain_files in MODEL_FILENAMES.values():
        for filename in chain_files.values():
            path = args.model_dir / filename
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "The following required model files were not found:\n  - "
            + "\n  - ".join(missing)
        )


def normalize_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def read_fasta(input_fasta: Path) -> pd.DataFrame:
    records = []
    seen = set()
    duplicate_ids = []

    for order, record in enumerate(SeqIO.parse(str(input_fasta), "fasta")):
        seq_id = record.id.strip()
        sequence = normalize_sequence(str(record.seq))

        if not seq_id:
            raise ValueError("A FASTA record has an empty sequence ID.")
        if seq_id in seen:
            duplicate_ids.append(seq_id)
            continue
        if not sequence:
            raise ValueError(f"Sequence '{seq_id}' is empty.")

        seen.add(seq_id)
        records.append(
            {
                "Input_Order": order,
                "ID": seq_id,
                "Input_Sequence": sequence,
            }
        )

    if duplicate_ids:
        examples = ", ".join(sorted(set(duplicate_ids))[:20])
        raise ValueError(
            f"Duplicate FASTA IDs were detected ({len(set(duplicate_ids))} unique IDs): "
            f"{examples}"
        )
    if not records:
        raise ValueError(f"No FASTA records were found in {input_fasta}.")

    return pd.DataFrame(records)


def process_one_sequence(item: tuple[str, str]) -> tuple[str, Optional[str], str, Optional[str]]:
    seq_id, sequence = item
    try:
        chain = Chain(sequence, scheme="imgt")
        processed_sequence = getattr(chain, "seq", None)
        if processed_sequence is None:
            processed_sequence = getattr(chain, "sequence", None)
        if processed_sequence is not None:
            processed_sequence = str(processed_sequence).replace("-", "")

        chain_type_raw = getattr(chain, "chain_type", None)
        if chain_type_raw == "K":
            chain_type = "Kappa"
        elif chain_type_raw == "L":
            chain_type = "Lambda"
        elif chain_type_raw == "H":
            chain_type = "Heavy"
        else:
            chain_type = "Unknown"

        if not processed_sequence:
            return seq_id, None, chain_type, "No valid variable-region sequence was identified."

        return seq_id, processed_sequence, chain_type, None
    except Exception as exc:  # AbNumber/ANARCI errors are retained per sequence.
        return seq_id, None, "Unknown", str(exc)


def process_sequences(df: pd.DataFrame, threads: int) -> pd.DataFrame:
    items = list(zip(df["ID"], df["Input_Sequence"]))
    results = {}

    if threads == 1:
        for item in items:
            result = process_one_sequence(item)
            results[result[0]] = result[1:]
    else:
        with ProcessPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(process_one_sequence, item): item[0] for item in items}
            for future in as_completed(futures):
                seq_id = futures[future]
                try:
                    result = future.result()
                    results[result[0]] = result[1:]
                except Exception as exc:
                    results[seq_id] = (None, "Unknown", str(exc))

    output = df.copy()
    output["Processed_Sequence"] = output["ID"].map(lambda x: results[x][0])
    output["Chain"] = output["ID"].map(lambda x: results[x][1])
    output["Processing_Error"] = output["ID"].map(lambda x: results[x][2])
    return output


def write_chain_fasta(df: pd.DataFrame, chain: str, output_fasta: Path) -> int:
    subset = df[
        (df["Chain"] == chain)
        & df["Processed_Sequence"].notna()
    ].sort_values("Input_Order")

    with output_fasta.open("w", encoding="utf-8") as handle:
        for row in subset.itertuples(index=False):
            handle.write(f">{row.ID}\n{row.Processed_Sequence}\n")

    return len(subset)


def run_ifeature(
    fasta_file: Path,
    output_tsv: Path,
    ifeature_script: Path,
    ifeature_python: str,
) -> None:
    command = [
        ifeature_python,
        str(ifeature_script),
        "--file",
        str(fasta_file),
        "--type",
        "CKSAAP",
        "--out",
        str(output_tsv),
    ]
    print("[iFeature]", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    if not output_tsv.is_file() or output_tsv.stat().st_size == 0:
        raise RuntimeError(f"iFeature did not create a valid output file: {output_tsv}")


def align_features(
    feature_tsv: Path,
    template_csv: Path,
) -> tuple[pd.Series, pd.DataFrame]:
    feature_df = pd.read_csv(feature_tsv, sep="\t")
    template_df = pd.read_csv(template_csv)

    if feature_df.empty:
        raise ValueError(f"No sequences were found in iFeature output: {feature_tsv}")
    if feature_df.shape[1] < 2:
        raise ValueError(f"Invalid iFeature output format: {feature_tsv}")
    if template_df.shape[1] < 2:
        raise ValueError(f"Invalid feature-template format: {template_csv}")

    id_column = feature_df.columns[0]

    # The first template column stores the sample identifier/class column used
    # during model development. Remaining columns are the fixed selected features.
    required_features = list(template_df.columns[1:])
    missing_features = [name for name in required_features if name not in feature_df.columns]
    if missing_features:
        preview = ", ".join(missing_features[:20])
        raise ValueError(
            f"{len(missing_features)} required features are missing from {feature_tsv}. "
            f"Examples: {preview}"
        )

    ids = feature_df[id_column].astype(str)
    x_test = feature_df.loc[:, required_features].apply(pd.to_numeric, errors="raise")

    if x_test.isna().any().any():
        raise ValueError(f"Missing values were detected after feature alignment: {feature_tsv}")

    return ids, x_test


def get_positive_probability(model, x_scaled: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x_scaled))
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("The model returned an invalid predict_proba matrix.")

        positive_index = 1
        classes = getattr(model, "classes_", None)
        if classes is not None:
            classes_list = list(classes)
            for positive_label in (1, "1", "AL", "Positive", True):
                if positive_label in classes_list:
                    positive_index = classes_list.index(positive_label)
                    break
        return probabilities[:, positive_index].astype(float)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x_scaled), dtype=float)
        scores = np.clip(scores, -700, 700)
        return 1.0 / (1.0 + np.exp(-scores))

    raise TypeError(
        "The loaded model provides neither predict_proba nor decision_function."
    )


def predict_chain(
    chain: str,
    feature_tsv: Path,
    model_dir: Path,
    threshold: float,
) -> pd.DataFrame:
    paths = MODEL_FILENAMES[chain]
    template_file = model_dir / paths["template"]
    model_file = model_dir / paths["model"]
    scaler_file = model_dir / paths["scaler"]

    ids, x_test = align_features(feature_tsv, template_file)

    scaler = joblib.load(scaler_file)
    model = joblib.load(model_file)

    x_scaled = scaler.transform(x_test)
    probability = get_positive_probability(model, x_scaled)
    label = (probability >= threshold).astype(int)

    return pd.DataFrame(
        {
            "ID": ids,
            "Chain": chain,
            "AL_Probability": probability,
            "Predicted_Label": label,
            "Prediction": np.where(label == 1, "AL", "non-AL"),
        }
    )


def main() -> int:
    args = parse_args()
    validate_args(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_file = args.output_dir / f"{args.prefix}_predictions.tsv"
    processing_file = args.output_dir / f"{args.prefix}_processed_sequences.tsv"
    failed_file = args.output_dir / f"{args.prefix}_failed_sequences.tsv"

    print(f"[1/4] Reading input FASTA: {args.input}", flush=True)
    input_df = read_fasta(args.input)

    print(
        f"[2/4] Extracting variable regions and assigning chain types "
        f"with {args.threads} worker(s)...",
        flush=True,
    )
    processed_df = process_sequences(input_df, args.threads)
    processed_df.to_csv(processing_file, sep="\t", index=False)

    failed_df = processed_df[
        processed_df["Processed_Sequence"].isna()
        | ~processed_df["Chain"].isin(["Kappa", "Lambda"])
    ].copy()
    if not failed_df.empty:
        failed_df.to_csv(failed_file, sep="\t", index=False)

    order_map = dict(zip(processed_df["ID"], processed_df["Input_Order"]))
    all_predictions = []

    if args.keep_intermediate:
        work_dir = args.output_dir / f"{args.prefix}_intermediate"
        work_dir.mkdir(parents=True, exist_ok=True)
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="alight_risk_")
        work_dir = Path(temp_context.name)

    try:
        print("[3/4] Extracting CKSAAP features and running fixed models...", flush=True)

        for chain in ("Kappa", "Lambda"):
            chain_key = chain.lower()
            chain_fasta = work_dir / f"{args.prefix}_{chain_key}.fasta"
            feature_tsv = work_dir / f"{args.prefix}_{chain_key}_CKSAAP.tsv"

            sequence_count = write_chain_fasta(processed_df, chain, chain_fasta)
            if sequence_count == 0:
                print(f"[skip] No valid {chain} sequences were identified.", flush=True)
                continue

            run_ifeature(
                fasta_file=chain_fasta,
                output_tsv=feature_tsv,
                ifeature_script=args.ifeature_script,
                ifeature_python=args.ifeature_python,
            )
            chain_predictions = predict_chain(
                chain=chain,
                feature_tsv=feature_tsv,
                model_dir=args.model_dir,
                threshold=args.threshold,
            )
            all_predictions.append(chain_predictions)

        if not all_predictions:
            raise RuntimeError(
                "No kappa or lambda sequence was successfully processed; "
                "no predictions were generated."
            )

        result_df = pd.concat(all_predictions, ignore_index=True)
        result_df["Input_Order"] = result_df["ID"].map(order_map)
        result_df = (
            result_df.sort_values("Input_Order")
            .drop(columns=["Input_Order"])
            .reset_index(drop=True)
        )
        result_df.to_csv(result_file, sep="\t", index=False, float_format="%.6f")

    finally:
        if temp_context is not None:
            temp_context.cleanup()

    total = len(processed_df)
    predicted = len(result_df)
    kappa_count = int((result_df["Chain"] == "Kappa").sum())
    lambda_count = int((result_df["Chain"] == "Lambda").sum())
    failed_count = total - predicted

    print("[4/4] Prediction completed.", flush=True)
    print(f"  Input sequences : {total}")
    print(f"  Predicted       : {predicted}")
    print(f"  Kappa           : {kappa_count}")
    print(f"  Lambda          : {lambda_count}")
    print(f"  Not predicted   : {failed_count}")
    print(f"  Results         : {result_file}")
    print(f"  Processing log  : {processing_file}")
    if not failed_df.empty:
        print(f"  Failed records  : {failed_file}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(
            f"[ERROR] External command failed with exit code {exc.returncode}: "
            f"{' '.join(map(str, exc.cmd))}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
