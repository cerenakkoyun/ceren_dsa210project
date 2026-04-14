# DSA 210 Term Project: Predicting Agricultural Crop Yield Across Turkish Provinces

## Overview

This project investigates the relationship between weather conditions and agricultural crop yields across Turkey's 81 provinces. Using province-level climate data from MGM (Turkish State Meteorological Service) and crop production statistics from TÜİK (Turkish Statistical Institute), I build a binary classification model to predict whether a province's yield will be **above average** or **below average** in a given year.

## Data Sources

| Source | Description | Variables |
|--------|-------------|-----------|
| **TÜİK** (Bitkisel Üretim İstatistikleri) | Province-level annual crop production data | Planted area, harvested area, production (tonnes), yield (kg/decare) |
| **MGM** (İl İklim İstatistikleri) | Province-level annual meteorological data | Avg. temperature (°C), total rainfall (mm), rainy days, relative humidity (%) |

## Project Structure

```
├── data/
│   └── turkey_agriculture_dataset.csv    # Merged & enriched dataset
├── src/
│   ├── data_collection.py                # Data compilation & feature engineering
│   └── eda_and_hypothesis_tests.py       # EDA visualizations & statistical tests
├── figures/                              # All generated plots (11 visualizations)
├── proposal_halfpage.docx                # Project proposal
├── AI_USAGE.md                           # AI tool usage disclosure (required)
├── .gitignore
├── requirements.txt
└── README.md
```

## Methodology

**Crops analyzed:** Wheat (Buğday), Barley (Arpa), Maize (Mısır), Sunflower (Ayçiçeği)

**Enrichment features derived:**
- Drought Stress Index (rainfall / temperature ratio)
- Rainfall deviation from provincial long-term normal (z-score)
- Seasonal temperature amplitude by region
- Year-over-year yield change rate
- Harvest ratio (harvested / planted area)

**Target variable:** Binary — 1 (above median yield) / 0 (below median yield), computed per crop type.

## Current Progress

- [x] Project proposal submitted
- [x] Data collection & feature engineering
- [x] Exploratory Data Analysis (EDA)
- [x] Hypothesis testing (t-test, ANOVA, Chi-square, Pearson correlation)
- [ ] Machine Learning models (Logistic Regression, Random Forest, XGBoost)
- [ ] Final report

## How to Reproduce

```bash
pip install -r requirements.txt
cd src
python data_collection.py
python eda_and_hypothesis_tests.py
```

## AI Tool Disclosure

This project used Claude (Anthropic) as an AI assistant for code generation and analysis support. Full details including specific prompts and outputs are documented in [`AI_USAGE.md`](AI_USAGE.md), as required by the course academic integrity policy.

## Tools & Libraries

Python 3.10+ — pandas, numpy, scipy, scikit-learn, xgboost, matplotlib, seaborn, geopandas
