# Medicaid Provider Anomaly Detection

## Setup

### 1. Install dependencies

Create and activate a virtual environment, then install requirements:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy the environment template and fill in your values:

```bash
cp .env.example .env
```

`.env.example`:
```
# Optional but recommended — required only if using the wandb logger.
# See usage for how to switch to the CSV logger instead.
WANDB_API_KEY=your_wandb_api_key_here
```

### 2. Set up Weights & Biases (optional)

wandb is not required — the CSV logger works out of the box with no account needed (see usage for details). If you do want wandb:

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Get your API key from **Settings → API keys**
3. Add it to your `.env` file as `WANDB_API_KEY`, or log in directly:

```bash
wandb login
```

### 3. Download datasets

Place all three files in `data/datasets/` under the exact filenames listed below.

---

#### Medicaid Provider Spending — `medicaid-provider-spending.csv`
**Source:** [opendata.hhs.gov/datasets/medicaid-provider-spending](https://opendata.hhs.gov/datasets/medicaid-provider-spending/)

Raw Medicaid billing data (~10 GB). One row per (provider, month, HCPCS code). Primary input for all feature engineering steps.

---

#### LEIE Exclusion Labels — `LEIElabels.csv`
**Source:** [oig.hhs.gov/exclusions/exclusions_list.asp](https://oig.hhs.gov/exclusions/exclusions_list.asp)

HHS Office of Inspector General List of Excluded Individuals/Entities — providers sanctioned for fraud or abuse. Used as **weak supervised labels**: LEIE-listed NPIs serve as positive fraud signals during model evaluation and label construction.

---

#### NPPES Provider Registry — `nppes.csv`
**Source:** [download.cms.gov/nppes/NPI_Files.html](https://download.cms.gov/nppes/NPI_Files.html)

National Plan and Provider Enumeration System full replacement file (~10 GB). Contains NPI, entity type (individual vs. organization), state, and taxonomy codes. Used to assign each provider a cohort based on `(state, entity_type)`.
