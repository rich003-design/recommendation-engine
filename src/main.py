"""
Main script for extracting GA4 data from BigQuery
"""

import argparse
from data_extraction import extract_ga4_data, save_data_to_csv
import os

def main():
    parser = argparse.ArgumentParser(description='Extract GA4 data from BigQuery')
    parser.add_argument('--project_id', required=True, help='Google Cloud project ID')
    parser.add_argument('--start_date', required=True, help='Start date (YYYYMMDD)')
    parser.add_argument('--end_date', required=True, help='End date (YYYYMMDD)')
    parser.add_argument('--credentials', default='config/secrets.json', 
                       help='Path to service account JSON file')
    parser.add_argument('--output', default='data/ga4_data.csv', 
                       help='Output CSV file path')
    parser.add_argument('--limit', type=int, help='Limit number of rows (for testing)')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Extract data
        df = extract_ga4_data(
            project_id=args.project_id,
            start_date=args.start_date,
            end_date=args.end_date,
            credentials_path=args.credentials,
            limit=args.limit
        )
        
        # Save data
        save_data_to_csv(df, args.output)
        
        print(f"Data extraction completed successfully!")
        print(f"Extracted {len(df)} rows")
        print(f"Saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
