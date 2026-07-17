# ALight-Risk

A sequence classifier for predicting the amyloidogenic risk of immunoglobulin light chains (AL). The model separates AL (amyloid-light-chain) positive sequences from non-AL sequences using kappa and lambda light-chain variable regions, and outputs an AL-risk probability for each input sequence.

This repository contains the raw data, the full training pipeline, and the final trained models together with a command-line predictor.

## Directory structure

```text
.
├── Data/                       # Raw input data
│   ├── AL-Base-202503-859.fasta        # AL-positive light-chain sequences (AL-Base, 859 entries)
│   ├── BCR_9AL_8HD.fasta               # BCR-derived light-chain sequences
│   └── OAS_McIntire_BM.fasta            # OAS (Observed Antibody Space) non-AL sequences
├── Pipeline/                  # Model training pipeline
│   ├── pipeline_config.example.env      # Example environment configuration for the pipelines
│   ├── kappa/                          # Kappa-chain training pipeline (self-contained)
│   │   ├── main_kappa.sh               # End-to-end orchestration script
│   │   ├── Step1_for_igBlast.py         # Extract Fv variable regions and build FASTA
│   │   ├── Step1.5_MutData.R            # Parse IgBLAST alignments into mutation tables
│   │   ├── Step2_SampleSplit.R          # Train / held-out test partitioning (seeded)
│   │   ├── Step2.5_BaseLineModel.py     # Baseline models (gene / mutation / length)
│   │   ├── Step3_iFeature.py            # Feature extraction with iFeature
│   │   ├── Step4_training.py            # Train and evaluate the final SVM models
│   │   ├── fasta2table.pl               # FASTA -> TSV helper
│   │   ├── fasta_to_oneline.py          # Multi-line FASTA -> one-line FASTA helper
│   │   └── filter_fasta.py             # Remove sequences with non-standard amino acids
│   └── lambda/                         # Lambda-chain training pipeline (self-contained)
│       ├── main_lambda.sh
│       ├── Step1_for_igBlast.py
│       ├── Step1.5_MutData.R
│       ├── Step2_SampleSplit.R
│       ├── Step2.5_BaseLineModel.py
│       ├── Step3_iFeature.py
│       ├── Step4_training.py
│       ├── fasta2table.pl
│       ├── fasta_to_oneline.py
│       └── filter_fasta.py
└── Prediction/                # Final trained models and command-line predictor
    ├── alight_risk_predict.py          # Predictor entry point
    ├── requirements_cli.txt            # Python dependencies for prediction
    ├── README_CLI.md                   # Detailed predictor usage
    └── Model_file/                     # Fixed feature templates, trained SVMs, fitted scalers
        ├── Kappa_trainset-1_CKSAAP_reduce.csv
        ├── kappa_1CKSAAP_SVM.pkl
        ├── kappa_1CKSAAP_scaler.pkl
        ├── Lambda_trainset-1_CKSAAP_reduce.csv
        ├── lambda_1CKSAAP_SVM.pkl
        └── lambda_1CKSAAP_scaler.pkl
```

## Data

The `Data/` directory holds the three raw FASTA files used to build the training set:

| File | Description |
| --- | --- |
| `AL-Base-202503-859.fasta` | AL-positive light-chain sequences curated from the AL-Base registry (used as positive samples). |
| `BCR_9AL_8HD.fasta` | BCR-repertoire light-chain sequences. |
| `OAS_McIntire_BM.fasta` | Non-AL light-chain sequences sampled from the Observed Antibody Space (OAS), used as negative samples. |

The training pipeline copies these into a working `1.Data/` directory (with `Pos/` and `Neg/` subfolders) at runtime; the original files in `Data/` are never modified.

## Pipeline

The `Pipeline/` directory contains two parallel, self-contained training pipelines — one per light-chain type (`kappa/` and `lambda/`). Each pipeline directory is treated as its own project root: its `main_*.sh` script orchestrates the full workflow and expects all Step scripts and helper files to sit next to it.

### Pipeline steps

1. **Preprocessing** (`Step1_for_igBlast.py`, `fasta2table.pl`, `fasta_to_oneline.py`, `filter_fasta.py`)
   Renames sequences, extracts IMGT-numbered Fv variable regions with [AbNumber](https://github.com/priestlabs/abnumber), removes sequences containing non-standard amino acids, and clusters non-AL sequences with [CD-HIT](http://weizhongli-lab.org/cd-hit/) at 90% identity.

2. **IgBLAST annotation & mutation tables** (`Step1.5_MutData.R`)
   Annotates the V-gene germline with [IgBLAST](https://ncbi.github.io/igblast/) and parses the alignments into per-position mutation tables.

3. **Sample split** (`Step2_SampleSplit.R`)
   Partitions sequences into a training set and a held-out test set using the provided random seed. Kappa uses 178/44 pos and balanced negatives; lambda uses 490/122 pos and balanced negatives.

4. **Baseline models** (`Step2.5_BaseLineModel.py`)
   Trains three simple baselines (V-gene identity, mutation load, sequence length) with seven classifiers (SVM, Random Forest, Gaussian Naive Bayes, KNN, Decision Tree, XGBoost, Logistic Regression) for comparison.

5. **Feature extraction** (`Step3_iFeature.py`)
   Computes a panel of sequence descriptors with [iFeature](https://github.com/Superzchen/iFeature) (AAC, APAAC, CKSAAGP, DPC, CKSAAP, DDE, GAAC, PAAC, GDPC, GTPC, Moran, Geary, NMBroto, CTDC, CTDT, CTDD, CTriad, KSCTriad, SOCNumber, QSOrder).

6. **Feature selection** (external MRMD3.0)
   Runs the [MRMD3.0](https://github.com/heshida01/mRMR) feature selector from a conda environment (`MRMD3` by default) to produce reduced feature matrices.

7. **Model training & evaluation** (`Step4_training.py`)
   Trains the seven classifiers on each reduced feature matrix, evaluates them with 10-fold cross-validation and on the held-out test set, and saves the trained models and scalers.

### Running a pipeline

Each pipeline is driven by a bash script and configured entirely through environment variables. Copy `pipeline_config.example.env` to an untracked `pipeline_config.env`, edit the paths, then run:

```bash
cd Pipeline/kappa          # or Pipeline/lambda
set -a
source pipeline_config.env
set +a
./main_kappa.sh            # or ./main_lambda.sh
```

The pipelines create the following working directories next to the script (these are git-ignored):

```text
1.Data/  2.Training_Data/  2.5_Baseline_model/  3.iFeature/
4.MRMD3.0/  5.MRMD_Result/  6.Model/  7.TrainingScore/  Result/  .checkpoints/
```

Interrupted runs can be resumed thanks to checkpoint files under `.checkpoints/`. To force a step to rerun, delete the corresponding `.checkpoints/<chain>/<step>.done` file.

### External dependencies for the pipeline

- Python 3 with `abnumber`, `biopython`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`
- R with `stringr`, `dplyr`
- Perl
- [CD-HIT](http://weizhongli-lab.org/cd-hit/)
- [IgBLAST](https://ncbi.github.io/igblast/) with a human V-germline database (`IGBLAST_GERMLINE_DB`)
- [iFeature](https://github.com/Superzchen/iFeature) (`iFeature.py`)
- [MRMD3.0](https://github.com/heshida01/mRMR) in a conda environment (`MRMD3` by default)
- conda

See `Pipeline/pipeline_config.example.env` for all configurable knobs.

## Prediction

The `Prediction/` directory contains the final trained models and a command-line predictor that classifies light-chain sequences as AL or non-AL.

### Quick start

```bash
conda create -n alight-risk python=3.9 -y
conda activate alight-risk
pip install -r Prediction/requirements_cli.txt

python Prediction/alight_risk_predict.py \
  --input your_sequences.fasta \
  --model-dir Prediction/Model_file \
  --ifeature-script /path/to/iFeature/iFeature.py \
  --output-dir result \
  --threads 8
```

### Outputs

```text
result/ALight-Risk_predictions.tsv          # ID, chain, AL probability, label, prediction
result/ALight-Risk_processed_sequences.tsv  # Per-sequence processing log
result/ALight-Risk_failed_sequences.tsv     # Sequences that could not be assigned (if any)
```

The default classification threshold is 0.5 and can be changed with `--threshold`. The predictor uses the fixed selected features, fitted scalers, and trained SVM models shipped in `Prediction/Model_file/`; it does not rerun feature selection or model training.

For the full predictor reference, see [Prediction/README_CLI.md](Prediction/README_CLI.md).

## Citation

-
