# Analysis scripts for: Subthalamic stimulation modulates working memory-related cortical dynamics in Parkinson's disease
Python and R analysis scripts for EEG and behavioral analyses of working-memory-related cortical dynamics in Parkinson's disease during subthalamic nucleus deep brain stimulation. 

This repository contains analysis scripts for the manuscript. Patient-level data and generated analysis outputs are not included in the public repository.

Core analysis code written by Marius Keute and Silvana Miranda Montenegro.

## Repository Layout

```text
.
├── main.py                         # Python analysis workflow
├── statistics.R                    # R mixed-model and plotting workflow
├── project_package/
│   └── project_functions/          # Reusable Python analysis helpers
├── data/                           # Local/private data, ignored by git
│   ├── metadata/                   # Measurement metadata CSVs
│   ├── raw/                        # Raw FIF files, if preprocessing is rerun
│   ├── preprocessed/               # Preprocessed FIF and epoch files
│   ├── processed/                  # Cached pickle/CSV outputs
│   └── r_inputs/                   # CSV exports consumed by statistics.R
└── results/                        # Local/generated outputs, ignored by git
    ├── figures/
    └── statistics/
```

## Data Placement

The code expects data to be placed locally using the structure above. At minimum, the metadata files should be placed in `data/metadata/`:

- `Analizable_info.csv`
- `Data_file_names.csv`
- `Stimulation_Info.csv`

If analysis steps are not rerun, cached outputs should be placed in `data/processed/`:

- `behScores.pkl`
- `p300potentialsinfo.pkl`
- `preprocessed_files_info_1.csv`
- `timefrequencydata.pkl`

Preprocessed FIF files should be placed in `data/preprocessed/`. Raw FIF files should be placed in `data/raw/` only when rerunning preprocessing.

## Running

Install the Python dependencies listed in `requirements.txt` and the R packages listed in `r-packages.txt`, then configure the `redo_*` flags at the top of `main.py`.

```bash
python main.py
Rscript statistics.R
```

`main.py` writes derived Python outputs to `data/processed/`, R input tables to `data/r_inputs/`, statistics tables to `results/statistics/`, and figures to `results/figures/`.
