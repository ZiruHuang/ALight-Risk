#!/usr/bin/env bash
# ALight-Risk kappa-chain training pipeline
#
# This script is intended to be run from the ALight-Risk repository root.
# User-specific software paths are configured through environment variables
# rather than hard-coded absolute paths.
#
# Required configuration:
#   IGBLAST_GERMLINE_DB=/path/to/human_germline_IGLV_IGKV
#
# Optional configuration examples:
#   PROJECT_ROOT=/path/to/ALight-Risk \
#   PYTHON_BIN=python3 RSCRIPT_BIN=Rscript \
#   IGBLASTP_BIN=igblastp CDHIT_BIN=cd-hit \
#   START_ROUND=1 END_ROUND=30 THREADS=8 \
#   ./main_kappa.sh
#
# The pipeline uses checkpoint files under .checkpoints/kappa so interrupted
# runs can be resumed. Delete the relevant checkpoint only when intentionally
# rerunning that step.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
cd "$PROJECT_ROOT"

# --------------------------- Configuration ---------------------------
CHAIN="Kappa"
CHAIN_KEY="kappa"
MRMD_SCRIPT="kappa_mrmd3_submodel_1.sh"
CDHIT_IDENTITY="${CDHIT_IDENTITY:-0.90}"

START_ROUND="${START_ROUND:-1}"
END_ROUND="${END_ROUND:-30}"
THREADS="${THREADS:-8}"
BASELINE_THREADS="${BASELINE_THREADS:-12}"
KEEP_FINAL_WORKDIR="${KEEP_FINAL_WORKDIR:-1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RSCRIPT_BIN="${RSCRIPT_BIN:-Rscript}"
PERL_BIN="${PERL_BIN:-perl}"
CDHIT_BIN="${CDHIT_BIN:-cd-hit}"
IGBLASTP_BIN="${IGBLASTP_BIN:-igblastp}"
CONDA_BIN="${CONDA_BIN:-conda}"
MRMD_CONDA_ENV="${MRMD_CONDA_ENV:-MRMD3}"
IGBLAST_GERMLINE_DB="${IGBLAST_GERMLINE_DB:-}"
IFEATURE_SCRIPT="${IFEATURE_SCRIPT:-}"

DATA_TEMPLATE_DIR="${DATA_TEMPLATE_DIR:-$PROJECT_ROOT/1.Data_bk}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/1.Data}"
MRMD_TEMPLATE_DIR="${MRMD_TEMPLATE_DIR:-$PROJECT_ROOT/4.MRMD3.0_bk}"
MRMD_WORK_DIR="${MRMD_WORK_DIR:-$PROJECT_ROOT/4.MRMD3.0}"
TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-$PROJECT_ROOT/2.Training_Data}"
IFEATURE_DIR="${IFEATURE_DIR:-$PROJECT_ROOT/3.iFeature}"
MRMD_RESULT_DIR="${MRMD_RESULT_DIR:-$PROJECT_ROOT/5.MRMD_Result}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/6.Model}"
TRAINING_SCORE_DIR="${TRAINING_SCORE_DIR:-$PROJECT_ROOT/7.TrainingScore}"
BASELINE_RESULT_DIR="${BASELINE_RESULT_DIR:-$PROJECT_ROOT/2.5_Baseline_model}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/Result}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/.checkpoints/$CHAIN_KEY}"

POS_FASTA="${POS_FASTA:-$DATA_DIR/Pos/AL-Base-202503-859.fasta}"
NEG_SOURCE_FASTA="${NEG_SOURCE_FASTA:-$DATA_DIR/Neg/2.OAS_seq_fv.fasta}"

POS_CLEAN_FASTA="$DATA_DIR/Pos/3.cleaned-Pos_seq_fv.fasta"
POS_FINAL_FASTA="$DATA_DIR/Pos/4.cdhit-Pos_seq_fv.fasta"
POS_ONELINE_FASTA="$DATA_DIR/Pos/4.cdhit-Pos_seq_fv_oneline.fasta"
NEG_ALL_FASTA="$DATA_DIR/Neg/2.All.Neg_seq_fv.fasta"
NEG_CLEAN_FASTA="$DATA_DIR/Neg/3.cleaned-Neg_seq_fv.fasta"
NEG_FINAL_FASTA="$DATA_DIR/Neg/4.cdhit-Neg_seq_fv.fasta"
NEG_ONELINE_FASTA="$DATA_DIR/Neg/4.cdhit-Neg_seq_fv_oneline.fasta"

mkdir -p "$CHECKPOINT_DIR" "$RESULT_DIR"

# ------------------------------ Helpers ------------------------------
log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
    printf '[ERROR] %s\n' "$*" >&2
    exit 1
}

trap 'status=$?; printf "[ERROR] Command failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR

checkpoint_file() {
    printf '%s/%s.done' "$CHECKPOINT_DIR" "$1"
}

is_complete() {
    [[ -s "$(checkpoint_file "$1")" ]]
}

mark_complete() {
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" > "$(checkpoint_file "$1")"
}

require_command() {
    local command_name="$1"
    if [[ "$command_name" == */* ]]; then
        [[ -x "$command_name" ]] || die "Executable not found or not executable: $command_name"
    else
        command -v "$command_name" >/dev/null 2>&1 || die "Required command not found in PATH: $command_name"
    fi
}

require_file() {
    [[ -f "$1" ]] || die "Required file not found: $1"
}

require_dir() {
    [[ -d "$1" ]] || die "Required directory not found: $1"
}

copy_template_if_missing() {
    local source_dir="$1"
    local target_dir="$2"
    if [[ ! -d "$target_dir" ]]; then
        require_dir "$source_dir"
        log "Creating $(basename "$target_dir") from $(basename "$source_dir")"
        cp -a "$source_dir" "$target_dir"
    fi
}

move_named_files() {
    local search_dir="$1"
    local name_pattern="$2"
    local destination="$3"
    local -a files=()

    [[ -d "$search_dir" ]] || die "Output directory not found: $search_dir"
    mkdir -p "$destination"
    mapfile -d '' files < <(find "$search_dir" -type f -name "$name_pattern" -print0)
    ((${#files[@]} > 0)) || die "No files matching '$name_pattern' were found under $search_dir"
    mv -- "${files[@]}" "$destination/"
}

copy_named_files() {
    local search_dir="$1"
    local name_pattern="$2"
    local destination="$3"
    local -a files=()

    [[ -d "$search_dir" ]] || die "Output directory not found: $search_dir"
    mkdir -p "$destination"
    mapfile -d '' files < <(find "$search_dir" -type f -name "$name_pattern" -print0)
    ((${#files[@]} > 0)) || die "No files matching '$name_pattern' were found under $search_dir"
    cp -a -- "${files[@]}" "$destination/"
}

reset_round_workdirs() {
    rm -rf "$TRAINING_DATA_DIR" "$IFEATURE_DIR" "$MRMD_WORK_DIR" \
           "$MRMD_RESULT_DIR" "$MODEL_DIR" "$TRAINING_SCORE_DIR"
    cp -a "$MRMD_TEMPLATE_DIR" "$MRMD_WORK_DIR"
}

validate_configuration() {
    [[ "$START_ROUND" =~ ^[0-9]+$ ]] || die "START_ROUND must be a positive integer"
    [[ "$END_ROUND" =~ ^[0-9]+$ ]] || die "END_ROUND must be a positive integer"
    ((START_ROUND >= 1 && END_ROUND >= START_ROUND)) || die "Invalid round range: $START_ROUND-$END_ROUND"
    [[ -n "$IGBLAST_GERMLINE_DB" ]] || die "Set IGBLAST_GERMLINE_DB to the IgBLAST V-germline database prefix"
    [[ -n "$IFEATURE_SCRIPT" ]] || die "Set IFEATURE_SCRIPT to the path of iFeature.py"
    require_file "$IFEATURE_SCRIPT"

    require_command "$PYTHON_BIN"
    require_command "$RSCRIPT_BIN"
    require_command "$PERL_BIN"
    require_command "$CDHIT_BIN"
    require_command "$IGBLASTP_BIN"
    require_command "$CONDA_BIN"

    require_file "$PROJECT_ROOT/fasta2table.pl"
    require_file "$PROJECT_ROOT/Step1_for_igBlast.py"
    require_file "$PROJECT_ROOT/filter_fasta.py"
    require_file "$PROJECT_ROOT/fasta_to_oneline.py"
    require_file "$PROJECT_ROOT/Step1.5_MutData.R"
    require_file "$PROJECT_ROOT/Step2_SampleSplit.R"
    require_file "$PROJECT_ROOT/Step2.5_BaseLineModel.py"
    require_file "$PROJECT_ROOT/Step3_iFeature.py"
    require_file "$PROJECT_ROOT/Step4_training.py"
    require_dir "$DATA_TEMPLATE_DIR"
    require_dir "$MRMD_TEMPLATE_DIR"
}

validate_configuration

# ------------------------- Step 1: preprocessing -------------------------
if ! is_complete "step1_preprocessing"; then
    log "Running sequence preprocessing"
    copy_template_if_missing "$DATA_TEMPLATE_DIR" "$DATA_DIR"
    copy_template_if_missing "$MRMD_TEMPLATE_DIR" "$MRMD_WORK_DIR"

    require_file "$POS_FASTA"
    sed -i 's/\r$//' "$POS_FASTA"

    "$PERL_BIN" "$PROJECT_ROOT/fasta2table.pl" "$POS_FASTA" > "$DATA_DIR/Pos/1.Pos_seq.tsv"
    "$PYTHON_BIN" "$PROJECT_ROOT/Step1_for_igBlast.py"

    require_file "$NEG_SOURCE_FASTA"
    cp -- "$NEG_SOURCE_FASTA" "$NEG_ALL_FASTA"
    log "Negative sequences before filtering: $(grep -c '^>' "$NEG_ALL_FASTA" || true)"

    "$PYTHON_BIN" "$PROJECT_ROOT/filter_fasta.py" "$NEG_ALL_FASTA" "$NEG_CLEAN_FASTA"
    "$CDHIT_BIN" \
        -i "$NEG_CLEAN_FASTA" \
        -o "$NEG_FINAL_FASTA" \
        -c "$CDHIT_IDENTITY" \
        -d 0

    # AL-positive sequences are intentionally retained without clustering.
    require_file "$POS_CLEAN_FASTA"
    cp -- "$POS_CLEAN_FASTA" "$POS_FINAL_FASTA"

    mark_complete "step1_preprocessing"
else
    log "Skipping sequence preprocessing (checkpoint found)"
fi

# --------------------------- Step 2: IgBLAST ---------------------------
if ! is_complete "step2_igblast"; then
    log "Running IgBLAST annotation"

    require_file "$POS_FINAL_FASTA"
    require_file "$NEG_FINAL_FASTA"

    "$IGBLASTP_BIN" \
        -germline_db_V "$IGBLAST_GERMLINE_DB" \
        -query "$POS_FINAL_FASTA" \
        -outfmt '7 qseqid sseqid pident length mismatch gapopen gaps qstart qend sstart send evalue bitscore' \
        -organism human \
        -out "$DATA_DIR/Pos/5.Pos_igblast.out" \
        -num_threads "$THREADS"
    awk '/^V/' "$DATA_DIR/Pos/5.Pos_igblast.out" > "$DATA_DIR/Pos/6.Pos_igblast_table.xls"

    "$IGBLASTP_BIN" \
        -germline_db_V "$IGBLAST_GERMLINE_DB" \
        -query "$NEG_FINAL_FASTA" \
        -outfmt '7 qseqid sseqid pident length mismatch gapopen gaps qstart qend sstart send evalue bitscore' \
        -organism human \
        -out "$DATA_DIR/Neg/5.Neg_igblast.out" \
        -num_threads "$THREADS"
    awk '/^V/' "$DATA_DIR/Neg/5.Neg_igblast.out" > "$DATA_DIR/Neg/6.Neg_igblast_table.xls"

    # One-line FASTA files are generated before the baseline IgBLAST runs.
    "$PYTHON_BIN" "$PROJECT_ROOT/fasta_to_oneline.py" "$NEG_FINAL_FASTA" > "$NEG_ONELINE_FASTA"
    "$PYTHON_BIN" "$PROJECT_ROOT/fasta_to_oneline.py" "$POS_FINAL_FASTA" > "$POS_ONELINE_FASTA"

    "$IGBLASTP_BIN" \
        -germline_db_V "$IGBLAST_GERMLINE_DB" \
        -query "$NEG_ONELINE_FASTA" \
        -organism human \
        -out "$DATA_DIR/Neg_seq_igblast.out" \
        -num_threads "$BASELINE_THREADS"

    "$IGBLASTP_BIN" \
        -germline_db_V "$IGBLAST_GERMLINE_DB" \
        -query "$POS_ONELINE_FASTA" \
        -organism human \
        -out "$DATA_DIR/Pos_seq_igblast.out" \
        -num_threads "$BASELINE_THREADS"

    "$RSCRIPT_BIN" "$PROJECT_ROOT/Step1.5_MutData.R"
    mark_complete "step2_igblast"
else
    log "Skipping IgBLAST annotation (checkpoint found)"
fi

# ---------------------- Steps 3-7: repeated rounds ----------------------
for ((round = START_ROUND; round <= END_ROUND; round++)); do
    log "===== $CHAIN round $round/$END_ROUND ====="

    if ! is_complete "round_${round}_sample_split"; then
        log "Generating training and held-out partitions"
        "$PYTHON_BIN" "$PROJECT_ROOT/fasta_to_oneline.py" "$POS_FINAL_FASTA" > "$POS_ONELINE_FASTA"
        "$PYTHON_BIN" "$PROJECT_ROOT/fasta_to_oneline.py" "$NEG_FINAL_FASTA" > "$NEG_ONELINE_FASTA"
        "$PERL_BIN" "$PROJECT_ROOT/fasta2table.pl" "$POS_ONELINE_FASTA" > "$DATA_DIR/Pos/4.cdhit-Pos_seq_fv.tsv"
        "$PERL_BIN" "$PROJECT_ROOT/fasta2table.pl" "$NEG_ONELINE_FASTA" > "$DATA_DIR/Neg/4.cdhit-Neg_seq_fv.tsv"
        mkdir -p "$TRAINING_DATA_DIR/$CHAIN"
        "$RSCRIPT_BIN" "$PROJECT_ROOT/Step2_SampleSplit.R" "$round"
        mark_complete "round_${round}_sample_split"
    else
        log "Skipping sample split for round $round"
    fi

    if ! is_complete "round_${round}_baseline"; then
        log "Training baseline models"
        "$PYTHON_BIN" "$PROJECT_ROOT/Step2.5_BaseLineModel.py"
        mark_complete "round_${round}_baseline"
    else
        log "Skipping baseline models for round $round"
    fi

    if ! is_complete "round_${round}_ifeature"; then
        log "Extracting iFeature descriptors"
        "$PYTHON_BIN" "$PROJECT_ROOT/Step3_iFeature.py"
        mark_complete "round_${round}_ifeature"
    else
        log "Skipping iFeature extraction for round $round"
    fi

    if ! is_complete "round_${round}_mrmd"; then
        log "Running MRMD feature selection"
        copy_template_if_missing "$MRMD_TEMPLATE_DIR" "$MRMD_WORK_DIR"
        require_file "$MRMD_WORK_DIR/$MRMD_SCRIPT"
        (
            cd "$MRMD_WORK_DIR"
            "$CONDA_BIN" run -n "$MRMD_CONDA_ENV" bash "$MRMD_SCRIPT"
        )

        mkdir -p "$MRMD_RESULT_DIR/$CHAIN/trainset_1"
        mapfile -d '' mrmd_outputs < <(find "$MRMD_WORK_DIR/Results" -maxdepth 1 -mindepth 1 -name 'Kappa_trainset-1*' -print0)
        ((${#mrmd_outputs[@]} > 0)) || die "No Kappa MRMD outputs were produced"
        mv -- "${mrmd_outputs[@]}" "$MRMD_RESULT_DIR/$CHAIN/trainset_1/"
        mark_complete "round_${round}_mrmd"
    else
        log "Skipping MRMD for round $round"
    fi

    if ! is_complete "round_${round}_training"; then
        log "Training the $CHAIN model"
        "$PYTHON_BIN" "$PROJECT_ROOT/Step4_training.py"
        mark_complete "round_${round}_training"
    else
        log "Skipping model training for round $round"
    fi

    if ! is_complete "round_${round}_archive"; then
        log "Archiving outputs for round $round"
        round_dir="$RESULT_DIR/Round-${round}"
        chain_result_dir="$round_dir/$CHAIN"
        all_train_dir="$chain_result_dir/All_TrainData"

        mkdir -p "$chain_result_dir" "$all_train_dir"
        rm -rf "$round_dir/Data"
        cp -a "$TRAINING_DATA_DIR" "$round_dir/Data"

        move_named_files "$MODEL_DIR/$CHAIN" '*results.csv' "$chain_result_dir"
        move_named_files "$MODEL_DIR/$CHAIN" '*mutil_model_test.txt' "$chain_result_dir"
        move_named_files "$TRAINING_SCORE_DIR/$CHAIN" '*_mutil_model_test.txt' "$all_train_dir"
        move_named_files "$MODEL_DIR/$CHAIN" '*prob.txt' "$chain_result_dir"
        move_named_files "$MODEL_DIR/$CHAIN" '*cv_probabilities.csv' "$chain_result_dir"
        copy_named_files "$MODEL_DIR/$CHAIN" '*.pkl' "$chain_result_dir"
        move_named_files "$TRAINING_SCORE_DIR/$CHAIN" '*prob.txt' "$all_train_dir"

        rm -rf "$round_dir/5.MRMD_Result" "$round_dir/Model" "$round_dir/2.5_Baseline_model"
        mv "$MRMD_RESULT_DIR" "$round_dir/5.MRMD_Result"
        find "$round_dir/5.MRMD_Result" -type f -name '*.png' -delete
        mv "$MODEL_DIR/$CHAIN/trainset_1" "$round_dir/Model"
        mv "$BASELINE_RESULT_DIR" "$round_dir/2.5_Baseline_model"

        mark_complete "round_${round}_archive"
    else
        log "Skipping output archiving for round $round"
    fi

    if ((round < END_ROUND)) || [[ "$KEEP_FINAL_WORKDIR" != "1" ]]; then
        log "Resetting temporary work directories"
        reset_round_workdirs
    else
        log "Keeping final-round work directories (KEEP_FINAL_WORKDIR=1)"
    fi

    log "===== $CHAIN round $round completed ====="
done

log "$CHAIN pipeline completed successfully"
