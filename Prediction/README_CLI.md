# ALight-Risk command-line prediction

## Expected model directory

The directory passed with `--model-dir` must contain:

```text
Model_file/
├── Kappa_trainset-1_CKSAAP_reduce.csv
├── kappa_1CKSAAP_SVM.pkl
├── kappa_1CKSAAP_scaler.pkl
├── Lambda_trainset-1_CKSAAP_reduce.csv
├── lambda_1CKSAAP_SVM.pkl
└── lambda_1CKSAAP_scaler.pkl
```

## Installation

```bash
conda create -n alight-risk python=3.9 -y
conda activate alight-risk
pip install -r requirements_cli.txt
```

Install or clone iFeature separately and record the path to `iFeature.py`.

## Run

```bash
chmod +x alight_risk_predict.py

./alight_risk_predict.py \
  --input example.fasta \
  --model-dir /path/to/Model_file \
  --ifeature-script /path/to/iFeature/iFeature.py \
  --output-dir result \
  --threads 8
```

Equivalent Python invocation:

```bash
python alight_risk_predict.py \
  -i example.fasta \
  --model-dir /path/to/Model_file \
  --ifeature-script /path/to/iFeature/iFeature.py \
  -o result
```

If iFeature must be executed with a different Python interpreter:

```bash
python alight_risk_predict.py \
  -i example.fasta \
  --model-dir /path/to/Model_file \
  --ifeature-script /path/to/iFeature/iFeature.py \
  --ifeature-python /path/to/python3.7 \
  -o result
```

## Main output

```text
result/ALight-Risk_predictions.tsv
```

Columns:

- `ID`: input FASTA sequence identifier
- `Chain`: Kappa or Lambda
- `AL_Probability`: probability assigned to the AL class
- `Predicted_Label`: 1 for AL and 0 for non-AL
- `Prediction`: AL or non-AL

The default classification threshold is 0.5 and can be changed with `--threshold`.

The program also writes a processing table for all submitted sequences and a separate
failure table when a sequence cannot be assigned to a valid kappa or lambda variable region.

## Important implementation detail

The command-line program uses the fixed selected features, fitted scalers, and trained
models supplied in `Model_file`. It does not rerun MRMD feature selection or model training
for submitted sequences.
