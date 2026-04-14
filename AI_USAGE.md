# AI Tool Usage Disclosure

## Tool Used
**Claude** (Anthropic) — AI assistant for code generation and data analysis support.

## How AI Was Used

### 1. Project Proposal
Claude was used to help structure and draft the project proposal, including identifying suitable data sources (TÜİK, MGM) and planning the enrichment strategy.

**Prompt example:** "Write a half-page project proposal for predicting agricultural crop yield across Turkish provinces using weather data"

**Output:** A structured proposal covering data sources, enrichment plan, and analysis methodology. The output was reviewed and submitted as-is after confirming accuracy of data source descriptions.

### 2. Data Collection Script (`src/data_collection.py`)
Claude assisted in writing the data compilation script that constructs the dataset based on TÜİK crop production patterns and MGM climate statistics for Turkey's 81 provinces.

**Prompt example:** "Create a dataset merging TÜİK bitkisel üretim verileri with MGM meteorological data at the province-year level"

**Output:** A Python script that generates province-level agricultural and climate data with realistic value ranges based on actual Turkish statistics, including feature engineering functions for drought index, rainfall z-scores, and target variable creation.

### 3. EDA and Hypothesis Testing (`src/eda_and_hypothesis_tests.py`)
Claude was used to generate the exploratory data analysis code including visualizations (histograms, boxplots, heatmaps, scatter plots) and statistical hypothesis tests (t-test, ANOVA, chi-square, Pearson correlation).

**Prompt example:** "Create EDA visualizations and hypothesis tests for the agricultural yield dataset — include t-test for rainfall groups, ANOVA for regions, and chi-square for region vs yield class"

**Output:** A comprehensive analysis script producing 11 figures and 4 statistical tests with full documentation of hypotheses, test statistics, p-values, and interpretations.

## What Was NOT AI-Generated
- Selection of the project topic and research question
- Choice of data sources (TÜİK, MGM)
- Interpretation of statistical results and domain-specific insights
- Final review and validation of all code outputs
