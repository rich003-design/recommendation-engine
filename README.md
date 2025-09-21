# Recommendation Engine - GA4 Data Extraction

This project extracts user interaction data from Google Analytics 4 (GA4) stored in BigQuery for building a recommendation engine.

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd recommendation-engine
2.	Set up virtual environment:
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3.	Configure Google Cloud credentials:
o	Create service account with BigQuery access
o	Download JSON key file
o	Place it in config/secrets.json
Usage
bash
# Extract data for specific date range
python main.py \
  --project_id your-project-id \
  --start_date 20201101 \
  --end_date 20201231 \
  --output data/raw/ga4_data.csv

# For testing with limited data
python main.py \
  --project_id your-project-id \
  --start_date 20201101 \
  --end_date 20201231 \
  --limit 1000 \
  --output data/raw/test_data.csv
Project Structure
text
recommendation-engine/
├── src/
│   └── data_extraction.py
├── config/
│   ├── settings.py
│   └── secrets.json (ignored by git)
├── data/
│   ├── raw/
│   └── processed/
├── tests/
│   └── test_data_extraction.py
├── main.py
├── requirements.txt
└── README.md
