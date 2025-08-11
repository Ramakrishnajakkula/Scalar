# Data Pipeline for AI Funnel
# Handles data ingestion, transformation, and storage for the AI-powered funnel

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
from typing import Dict, List, Any, Optional, Union

class DataPipeline:
    """
    Data pipeline for processing and preparing lead data for the AI funnel.
    This component handles data ingestion, cleaning, feature engineering,
    and storage for use by other components in the funnel.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data pipeline with configuration.
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            Configuration parameters for the pipeline
        """
        self.config = config or {
            "storage_path": "./data/",
            "log_level": "INFO",
            "required_fields": ["id", "email", "name"],
            "optional_fields": ["phone", "occupation", "interests", "source"],
            "enrichment_enabled": True
        }
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.get("log_level", "INFO")),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='data_pipeline.log'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Data pipeline initialized with configuration")
    
    def ingest_leads_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Ingest lead data from a CSV file.
        
        Parameters
        ----------
        file_path : str
            Path to the CSV file containing lead data
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing the lead data
        """
        self.logger.info(f"Ingesting leads from CSV: {file_path}")
        
        try:
            # Read CSV file into DataFrame
            df = pd.read_csv(file_path)
            
            # Log basic information about the data
            self.logger.info(f"Ingested {len(df)} leads with {len(df.columns)} attributes")
            
            # Validate required fields
            missing_fields = [field for field in self.config["required_fields"] 
                             if field not in df.columns]
            
            if missing_fields:
                self.logger.warning(f"Missing required fields: {', '.join(missing_fields)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error ingesting leads from CSV: {str(e)}")
            raise
    
    def ingest_leads_from_json(self, file_path: str) -> pd.DataFrame:
        """
        Ingest lead data from a JSON file.
        
        Parameters
        ----------
        file_path : str
            Path to the JSON file containing lead data
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing the lead data
        """
        self.logger.info(f"Ingesting leads from JSON: {file_path}")
        
        try:
            # Read JSON file
            with open(file_path, 'r') as f:
                leads_data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(leads_data)
            
            # Log basic information
            self.logger.info(f"Ingested {len(df)} leads with {len(df.columns)} attributes")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error ingesting leads from JSON: {str(e)}")
            raise
    
    def ingest_leads_from_api(self, api_url: str, headers: Dict[str, str] = None) -> pd.DataFrame:
        """
        Ingest lead data from an API endpoint.
        
        Parameters
        ----------
        api_url : str
            URL of the API endpoint
        headers : Dict[str, str], optional
            Headers for the API request
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing the lead data
        """
        self.logger.info(f"Ingesting leads from API: {api_url}")
        
        try:
            # This would use requests in a real implementation
            # For the prototype, we'll simulate API data
            
            self.logger.info("Simulating API data ingestion")
            
            # Simulated lead data
            leads_data = [
                {
                    "id": f"lead_{i:03d}",
                    "name": f"Test User {i}",
                    "email": f"user{i}@example.com",
                    "engagement_score": np.random.randint(1, 100),
                    "days_since_last_interaction": np.random.randint(30, 365),
                    "source": np.random.choice(["website", "referral", "ad", "webinar"]),
                    "interests": np.random.choice(["data science", "web dev", "mobile dev", "cloud", "ai"], 
                                                size=np.random.randint(1, 3)).tolist(),
                    "career_change_signal": np.random.choice([True, False], p=[0.3, 0.7]),
                    "price_sensitivity": np.random.choice(["low", "medium", "high"]),
                    "last_course_viewed": np.random.choice(["DS101", "WD201", "ML301", "AI401", None], p=[0.2, 0.2, 0.2, 0.2, 0.2])
                }
                for i in range(100)
            ]
            
            df = pd.DataFrame(leads_data)
            
            self.logger.info(f"Ingested {len(df)} leads with {len(df.columns)} attributes")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error ingesting leads from API: {str(e)}")
            raise
    
    def clean_lead_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess lead data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing raw lead data
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing cleaned lead data
        """
        self.logger.info("Cleaning lead data")
        
        try:
            # Make a copy to avoid modifying the original
            cleaned_df = df.copy()
            
            # Remove duplicates based on ID or email
            if "id" in cleaned_df.columns:
                cleaned_df.drop_duplicates(subset=["id"], keep="first", inplace=True)
            elif "email" in cleaned_df.columns:
                cleaned_df.drop_duplicates(subset=["email"], keep="first", inplace=True)
            
            # Fill missing values
            if "engagement_score" in cleaned_df.columns:
                cleaned_df["engagement_score"].fillna(0, inplace=True)
            
            if "days_since_last_interaction" in cleaned_df.columns:
                cleaned_df["days_since_last_interaction"].fillna(365, inplace=True)
            
            # Normalize text fields
            if "name" in cleaned_df.columns:
                cleaned_df["name"] = cleaned_df["name"].str.strip().str.title()
            
            if "email" in cleaned_df.columns:
                cleaned_df["email"] = cleaned_df["email"].str.strip().str.lower()
            
            # Log cleaning results
            self.logger.info(f"Data cleaning complete. {len(df) - len(cleaned_df)} duplicates removed")
            
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Error cleaning lead data: {str(e)}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for the lead data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing cleaned lead data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with engineered features
        """
        self.logger.info("Engineering features")
        
        try:
            # Make a copy to avoid modifying the original
            enhanced_df = df.copy()
            
            # Calculate recency score (higher score for more recent interaction)
            if "days_since_last_interaction" in enhanced_df.columns:
                enhanced_df["recency_score"] = 100 - np.minimum(enhanced_df["days_since_last_interaction"] / 3.65, 100)
            
            # Create engagement categories
            if "engagement_score" in enhanced_df.columns:
                enhanced_df["engagement_category"] = pd.cut(
                    enhanced_df["engagement_score"],
                    bins=[0, 20, 40, 60, 80, 100],
                    labels=["very_low", "low", "medium", "high", "very_high"]
                )
            
            # Extract domain from email
            if "email" in enhanced_df.columns:
                enhanced_df["email_domain"] = enhanced_df["email"].str.split("@").str[1]
                
                # Identify corporate vs personal emails
                personal_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]
                enhanced_df["email_type"] = enhanced_df["email_domain"].apply(
                    lambda x: "personal" if x in personal_domains else "corporate"
                )
            
            # Convert interests list to individual boolean features
            if "interests" in enhanced_df.columns and isinstance(enhanced_df["interests"].iloc[0], list):
                # Get unique interests
                all_interests = set()
                for interests_list in enhanced_df["interests"]:
                    if isinstance(interests_list, list):
                        all_interests.update(interests_list)
                
                # Create boolean features
                for interest in all_interests:
                    interest_col = f"interest_{interest.replace(' ', '_')}"
                    enhanced_df[interest_col] = enhanced_df["interests"].apply(
                        lambda x: interest in x if isinstance(x, list) else False
                    )
            
            # Calculate potential score based on engagement, recency, and other factors
            score_columns = []
            if "engagement_score" in enhanced_df.columns:
                score_columns.append("engagement_score")
            if "recency_score" in enhanced_df.columns:
                score_columns.append("recency_score")
                
            if score_columns:
                enhanced_df["potential_score"] = enhanced_df[score_columns].mean(axis=1)
                
                if "career_change_signal" in enhanced_df.columns:
                    # Boost score for career changers
                    enhanced_df.loc[enhanced_df["career_change_signal"] == True, "potential_score"] *= 1.2
                
                # Cap at 100
                enhanced_df["potential_score"] = np.minimum(enhanced_df["potential_score"], 100)
            
            # Log feature engineering results
            self.logger.info(f"Feature engineering complete. {len(enhanced_df.columns) - len(df.columns)} new features added")
            
            return enhanced_df
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {str(e)}")
            raise
    
    def export_to_json(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Export lead data to a JSON file.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing lead data
        file_path : str
            Path to the output JSON file
        """
        self.logger.info(f"Exporting {len(df)} leads to JSON: {file_path}")
        
        try:
            # Convert DataFrame to list of records
            records = df.to_dict(orient="records")
            
            # Write to JSON file
            with open(file_path, 'w') as f:
                json.dump(records, f, indent=2)
            
            self.logger.info(f"Successfully exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {str(e)}")
            raise
    
    def export_to_csv(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Export lead data to a CSV file.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing lead data
        file_path : str
            Path to the output CSV file
        """
        self.logger.info(f"Exporting {len(df)} leads to CSV: {file_path}")
        
        try:
            # Write to CSV file
            df.to_csv(file_path, index=False)
            
            self.logger.info(f"Successfully exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {str(e)}")
            raise
    
    def process_leads(self, source_type: str, source_path: str, output_path: str = None, 
                      output_format: str = "json") -> pd.DataFrame:
        """
        Process leads from source to output, performing all pipeline steps.
        
        Parameters
        ----------
        source_type : str
            Type of source ('csv', 'json', or 'api')
        source_path : str
            Path to the source file or API URL
        output_path : str, optional
            Path for the output file
        output_format : str, optional
            Format for the output file ('json' or 'csv')
            
        Returns
        -------
        pd.DataFrame
            Processed DataFrame containing lead data
        """
        self.logger.info(f"Processing leads from {source_type} source: {source_path}")
        
        try:
            # Ingest data from source
            if source_type.lower() == "csv":
                df = self.ingest_leads_from_csv(source_path)
            elif source_type.lower() == "json":
                df = self.ingest_leads_from_json(source_path)
            elif source_type.lower() == "api":
                df = self.ingest_leads_from_api(source_path)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            # Clean data
            df = self.clean_lead_data(df)
            
            # Engineer features
            df = self.engineer_features(df)
            
            # Export data if output path is provided
            if output_path:
                if output_format.lower() == "json":
                    self.export_to_json(df, output_path)
                elif output_format.lower() == "csv":
                    self.export_to_csv(df, output_path)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")
            
            self.logger.info(f"Lead processing complete. Final dataset has {len(df)} leads and {len(df.columns)} attributes")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing leads: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Process leads from an API (simulated)
    processed_leads = pipeline.process_leads(
        source_type="api",
        source_path="https://api.example.com/leads",
        output_path="./processed_leads.json"
    )
    
    # Show first few leads
    print(processed_leads.head())
    
    # Show summary statistics
    print("\nSummary Statistics:")
    print(processed_leads.describe())
