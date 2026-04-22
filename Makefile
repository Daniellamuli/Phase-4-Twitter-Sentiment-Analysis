# Makefile
# Run from the project root directory.
#
# Usage:
#   make run        — full end-to-end pipeline
#   make preprocess — preprocessing only
#   make test       — quick smoke test (loads data, cleans one tweet)
#   make clean      — remove generated figures and model files

.PHONY: run preprocess test clean

## Run the full end-to-end pipeline (preprocessing → training → evaluation)
run:
	python run_pipeline.py

## Run preprocessing only and save cleaned_tweets.csv
preprocess:
	python -c "from src.preprocess import run_preprocessing_pipeline; run_preprocessing_pipeline(save_output=True)"

## Quick smoke test — verifies data loads and text cleaning works
test:
	python -c "\
from src.load_data import load_and_prepare_data; \
from src.clean_text import clean_text; \
df = load_and_prepare_data(); \
sample = clean_text(df['tweet'].iloc[0]); \
print('Smoke test passed ✅'); \
print('Sample cleaned tweet:', sample[:80])"

## Remove generated output files
clean:
	rm -f figures/*.png
	rm -f models/*.pkl
	rm -f data/processed/*.csv
	@echo "Cleaned generated files ✅"
