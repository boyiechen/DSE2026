# Kaggle submission workflow

Run every command below from the repository root:

```bash
cd /Users/bc3139/repo/summer2026/DSE2026
```

Install or refresh the project environment if needed:

```bash
uv sync
```

## 1. Generate both submission files

### Practicum 1

```bash
uv run python practicum/submission/generate_practicum1_submission.py
```

Following the TA's correction, the authoritative file is
`practicum/practicum1/data/submission_template.csv`: it has 51 prediction rows
using the S1/S2/S3 naming scheme. The earlier 31-row rejection was caused by
the Kaggle scorer, not by this template.

The script estimates the S1, S2, and S3 panels, solves the three S3
counterfactuals, validates every ID and its order against that template, and
writes:

```text
practicum/submission/practicum1_submission.csv
```

The script must end with `SUCCESS` and report 51 validated rows before the CSV
is submitted. The intended final run uses the script defaults: 256 scrambled
Sobol expectation draws and 300 draws with a 6,000-state anchor for each
fixed-point counterfactual.

If the official template is stored elsewhere, pass it explicitly:

```bash
uv run python practicum/submission/generate_practicum1_submission.py \
  --template /path/to/submission_template.csv
```

### Practicum 2

```bash
uv run python practicum/submission/generate_practicum2_submission.py
```

This runs the complete structural estimation, 500 bootstrap replications,
the risk-aversion counterfactual, validates all 19 rows against the official
sample submission, and writes:

```text
practicum/submission/practicum2_submission.csv
```

The script must report `500/500` usable bootstrap fits and end with `SUCCESS`.
Do not reduce `--bootstrap-reps` for the final submission.

## 2. Submit with the Kaggle CLI

The commands below use the positional competition-slug syntax documented by
the project's Kaggle CLI 2.2.4. They upload only the generated CSV files;
the notebooks, Python scripts, parquet data, and zero-filled templates are
not submitted.

### Submit Practicum 1

```bash
uv run kaggle competitions submit dse2026-practicum-1 \
  -f "$(pwd)/practicum/submission/practicum1_submission.csv" \
  -m "S1-S2-S3 structural estimates and S3 counterfactuals"
```

Check its status:

```bash
uv run kaggle competitions submissions dse2026-practicum-1
```

If Kaggle still says that 31 rows were expected, do not truncate or rename the
generated CSV: that means the corrected scorer has not yet been deployed.
The official template and the validated output both require all 51 rows.

### Submit Practicum 2

```bash
uv run kaggle competitions submit dse2026-practicum-2 \
  -f "$(pwd)/practicum/submission/practicum2_submission.csv" \
  -m "Structural estimates, bootstrap standard errors, and counterfactuals"
```

Check its status:

```bash
uv run kaggle competitions submissions dse2026-practicum-2
```

## Files that should not be submitted

- The current `practicum/practicum1/data/sample_submission.csv` is the obsolete
  36-row A/B/C format. The generator deliberately does not use it.
- `practicum/practicum1/data/submission_template.csv` is the correct 51-row
  schema, but it contains placeholder zeros. Submit the generated file under
  `practicum/submission/`, not the template itself.
- `practicum/practicum2/data/sample_submission.csv` and
  `practicum/practicum2/code/sample_submission.csv` contain placeholder
  zeros.

If Kaggle reports an authentication or rules error, authenticate the CLI and
accept the competition rules in the Kaggle web interface, then rerun only the
corresponding `submit` command. The generated CSV does not need to be rebuilt
for an authentication failure.
