from google.cloud import bigquery
import pandas as pd
import os
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_bigquery_client(credentials_path: Optional[str] = None) -> bigquery.Client:
    """
    Set up BigQuery client with optional credentials path
    
    Args:
        credentials_path: Path to service account JSON file
    
    Returns:
        bigquery.Client: Configured BigQuery client
    """
    if credentials_path and os.path.exists(credentials_path):
        client = bigquery.Client.from_service_account_json(credentials_path)
    else:
        # Use default credentials (for Cloud Run/Cloud Functions)
        client = bigquery.Client()
    
    return client
def extract_ga4_data(project_id: str, 
                    start_date: str, 
                    end_date: str,
                    credentials_path: Optional[str] = None,
                    limit: Optional[int] = None) -> pd.DataFrame:
    """
    Extract view_item events from GA4 dataset in BigQuery
    
    Args:
        project_id: Google Cloud project ID
        start_date: Start date in format YYYYMMDD
        end_date: End date in format YYYYMMDD
        credentials_path: Path to service account JSON file
        limit: Maximum number of rows to return (for testing)
    
    Returns:
        pd.DataFrame: DataFrame containing view_item events
    """
    
    client = setup_bigquery_client(credentials_path)
    
    query = """
    WITH filtered_events AS (
        SELECT 
            user_pseudo_id,
            event_date,
            event_timestamp,
            event_name,
            (SELECT ep.value.string_value FROM UNNEST(event_params) AS ep WHERE ep.key = "page_title") AS page_title,
            (SELECT ep.value.string_value FROM UNNEST(event_params) AS ep WHERE ep.key = "item_id") AS item_id,
            (SELECT ep.value.string_value FROM UNNEST(event_params) AS ep WHERE ep.key = "item_name") AS item_name,
            (SELECT ep.value.string_value FROM UNNEST(event_params) AS ep WHERE ep.key = "item_category") AS item_category,
            (SELECT ep.value.int_value FROM UNNEST(event_params) AS ep WHERE ep.key = "engagement_time_msec") AS engagement_time_msec,
            (SELECT ep.value.int_value FROM UNNEST(event_params) AS ep WHERE ep.key = "value") AS value,
            geo.continent,
            geo.country,
            geo.region,
            geo.city,
            device.category AS device_category,
            device.mobile_brand_name,
            device.mobile_model_name,
            device.operating_system,
            traffic_source.name AS traffic_source,
            traffic_source.medium AS traffic_medium,
            traffic_source.source AS traffic_source_name
        FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
        WHERE event_name = 'view_item'
        AND _TABLE_SUFFIX BETWEEN @start_date AND @end_date
    )
    SELECT 
        *,
        TIMESTAMP_MICROS(event_timestamp) AS event_datetime
    FROM filtered_events
    WHERE page_title IS NOT NULL
    ORDER BY event_timestamp
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "STRING", start_date),
            bigquery.ScalarQueryParameter("end_date", "STRING", end_date)
        ]
    )
    
    try:
        logger.info(f"Extracting GA4 data from {start_date} to {end_date}")
        df = client.query(query, job_config=job_config).to_dataframe()
        logger.info(f"Successfully extracted {len(df)} view_item events")
        
        # Basic data validation
        if len(df) == 0:
            logger.warning("No data found for the specified date range")
        
        return df
        
    except Exception as e:
        logger.error(f"Error extracting GA4 data: {str(e)}")
        raise

def save_data_to_csv(df: pd.DataFrame, filename: str) -> None:
    """
    Save DataFrame to CSV file
    
    Args:
        df: DataFrame to save
        filename: Output filename
    """
    df.to_csv(filename, index=False)
    logger.info(f"Data saved to {filename}")

def load_data_from_csv(filename: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV file
    
    Args:
        filename: Input filename
    
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        raise FileNotFoundError(f"File {filename} not found")

