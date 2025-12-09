# 02805_climate_conv

Main analysis lives in `FINAL.ipynb`, an explainer-style notebook that walks through how we map and interpret the climate conversation on X/Twitter using hashtags, network analysis, communities, and text signals.

## What the notebook does
- Loads cleaned tweet data (`data/cleaned_twitter_embedded_data_hashtags_fixed.csv`) and a hashtag co-occurrence network (`networks/hashtag_cooccurrence_network.gexf`), standardizes hashtags, and reports basic stats (tweet counts, unique tags, locations, date span).
- Describes the hashtag network: degree distribution, density, connected components, assortativity, and comparison to random graphs; plots a ForceAtlas2 layout to visualize structure.
- Runs Louvain community detection, then summarizes the top communities with frequency tables and word clouds to reveal discussion themes.
- Examines emotional tone with a labeled emotion dataset, plotting overall distributions and weekly trends, and heatmaps of dominant emotions per community.
- Looks at geography: location–community heatmaps and a country-level map colored by dominant community.
- Cleans tweet text, extracts TF-IDF bigrams per community, and builds TF-IDF word clouds for both hashtags and tweet words for reporting visuals.

## Running it
1) Install dependencies (Python):
```
pip install -r requirements.txt
```
2) Open and run `FINAL.ipynb` (Jupyter/VS Code). Data paths in the notebook are workspace-relative; ensure the `data/` and `networks/` folders exist with the expected CSV/GEXF files.

## Data prep utilities
- `help_files/get_huggingface.py` downloads source data.
- `help_files/prepare.py` cleans and harmonizes raw data.
- `make_files/make_hashtag_network.py` builds the hashtag co-occurrence network used in the notebook.
