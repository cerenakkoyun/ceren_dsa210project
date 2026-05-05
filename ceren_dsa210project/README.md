# dsa210project

## Predicting Agricultural Crop Yield Across Turkish Provinces Using Weather and Environmental Data

### Project Overview
This project investigates the relationship between weather conditions and agricultural crop yields across Turkey's 81 provinces. Using province-level climate data from MGM (Turkish State Meteorological Service) and crop production statistics from TÜİK (Turkish Statistical Institute), I build a binary classification model to predict whether a province's yield will be **above average** or **below average** in a given year.

### Data Sources
- **TÜİK** — Bitkisel Üretim İstatistikleri (Crop Production Statistics): province-level planted area, harvested area, production, yield
- **MGM** — İl İklim İstatistikleri (Provincial Climate Statistics): avg. temperature, rainfall, humidity, rainy days

### Repository Structure
```
├── DATA/
│   └── turkey_agriculture_dataset.csv
├── EDA/
│   ├── eda.ipynb
│   └── *.png (8 visualizations)
├── HypothesisTesting/
│   ├── hypothesis_tests.ipynb
│   └── *.png (3 visualizations)
├── MachineLearning/
│   ├── ml_classification.ipynb
│   └── *.png (4 visualizations)
├── dataCleaning.ipynb
├── AI_USAGE.md
├── proposal.pdf
└── README.md
```

### Methodology
**Crops:** Wheat, Barley, Maize, Sunflower  
**Dataset:** 4,212 observations — 81 provinces × 4 crops × 13 years (2012–2024)  
**Target:** Binary classification — High yield (1) vs Low yield (0) based on median threshold per crop

### Current Progress
- [x] Proposal submitted
- [x] Data collection & cleaning (`dataCleaning.ipynb`)
- [x] Exploratory Data Analysis (`EDA/eda.ipynb`)
- [x] Hypothesis Testing (`HypothesisTesting/hypothesis_tests.ipynb`)
- [x] Machine Learning models (`MachineLearning/ml_classification.ipynb`)
- [ ] Final report

### How to Run
```bash
pip install pandas numpy scipy scikit-learn xgboost matplotlib seaborn
```
Run notebooks in order: `dataCleaning.ipynb` → `EDA/eda.ipynb` → `HypothesisTesting/hypothesis_tests.ipynb` → `MachineLearning/ml_classification.ipynb`

### AI Tool Disclosure
This project used Claude (Anthropic) as an AI assistant for code generation and analysis support, as permitted by the course policy. AI was used for structuring analysis scripts and generating visualizations. Topic selection, data source decisions, and result interpretation were done independently.
