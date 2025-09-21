import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.data_extraction import extract_ga4_data

class TestDataExtraction(unittest.TestCase):
    
    @patch('src.data_extraction.bigquery.Client')
    def test_extract_ga4_data(self, mock_client):
        # Mock the BigQuery client and query result
        mock_query_result = MagicMock()
        mock_query_result.to_dataframe.return_value = pd.DataFrame({
            'user_pseudo_id': ['user1', 'user2'],
            'event_date': ['20201201', '20251202'],
            'page_title': ['Product A', 'Product B']
        })
        
        mock_client_instance = MagicMock()
        mock_client_instance.query.return_value = mock_query_result
        mock_client.from_service_account_json.return_value = mock_client_instance
        
        # Test the function
        df = extract_ga4_data(
            project_id="recommendationengine-472815",
            start_date="20201201",
            end_date="20251202",
            credentials_path="test_credentials.json",
            limit=100
        )
        
        self.assertEqual(len(df), 2)
        self.assertIn('user_pseudo_id', df.columns)
        self.assertIn('page_title', df.columns)

if __name__ == '__main__':
    unittest.main()
