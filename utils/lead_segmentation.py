# Lead Segmentation Model
# This module implements the AI segmentation algorithm for cold leads

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

class LeadSegmentationModel:
    """
    AI model that segments cold leads based on their characteristics and behaviors.
    Uses K-means clustering with automated optimal cluster selection.
    """
    
    def __init__(self, n_clusters=None, random_state=42):
        """
        Initialize the segmentation model.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters to use. If None, will be determined automatically.
        random_state : int
            Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = None
        
    def preprocess_data(self, data):
        """
        Preprocess the lead data for clustering.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw lead data.
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed data.
        """
        # Store feature names
        self.feature_names = data.columns.tolist()
        
        # Handle missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        
        # Fill missing values
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
        
        # One-hot encode categorical features
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        
        # Scale numeric features
        data_scaled = self.scaler.fit_transform(data)
        
        return data_scaled, data.columns.tolist()
    
    def find_optimal_clusters(self, data, max_clusters=10):
        """
        Find the optimal number of clusters using silhouette score.
        
        Parameters:
        -----------
        data : array-like
            Preprocessed data.
        max_clusters : int
            Maximum number of clusters to try.
            
        Returns:
        --------
        int
            Optimal number of clusters.
        """
        silhouette_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg}")
        
        # Return the number of clusters with highest silhouette score
        optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because we start from 2 clusters
        print(f"Optimal number of clusters: {optimal_clusters}")
        return optimal_clusters
    
    def fit(self, data):
        """
        Fit the segmentation model to the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Lead data.
            
        Returns:
        --------
        self
        """
        # Preprocess the data
        data_scaled, feature_names = self.preprocess_data(data)
        
        # Find the optimal number of clusters if not provided
        if self.n_clusters is None:
            self.n_clusters = self.find_optimal_clusters(data_scaled)
        
        # Perform dimensionality reduction for visualization
        self.pca = PCA(n_components=2)
        pca_result = self.pca.fit_transform(data_scaled)
        
        # Fit K-means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans.fit(data_scaled)
        
        return self
    
    def predict(self, data):
        """
        Assign clusters to new data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            New lead data.
            
        Returns:
        --------
        array-like
            Cluster assignments.
        """
        # Preprocess the data
        data_scaled, _ = self.preprocess_data(data)
        
        # Predict clusters
        return self.kmeans.predict(data_scaled)
    
    def visualize_clusters(self, data):
        """
        Visualize the clusters in 2D space.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Lead data.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with cluster visualization.
        """
        # Preprocess the data
        data_scaled, _ = self.preprocess_data(data)
        
        # Apply PCA
        pca_result = self.pca.transform(data_scaled)
        
        # Get cluster assignments
        clusters = self.kmeans.predict(data_scaled)
        
        # Create a DataFrame for visualization
        df_plot = pd.DataFrame({
            'PCA1': pca_result[:, 0],
            'PCA2': pca_result[:, 1],
            'Cluster': clusters
        })
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(data=df_plot, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', ax=ax)
        plt.title('Lead Segments Visualization')
        
        return fig
    
    def get_cluster_characteristics(self, data):
        """
        Get the characteristics of each cluster.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Original lead data.
            
        Returns:
        --------
        dict
            Characteristics of each cluster.
        """
        # Preprocess the data
        data_scaled, feature_names = self.preprocess_data(data)
        
        # Get cluster assignments
        clusters = self.kmeans.predict(data_scaled)
        
        # Add cluster assignments to the original data
        data_with_clusters = data.copy()
        data_with_clusters['Cluster'] = clusters
        
        # Get characteristics for each cluster
        cluster_characteristics = {}
        
        for cluster in range(self.n_clusters):
            cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
            
            # Get the mean values for numeric features
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            cluster_means = cluster_data[numeric_cols].mean()
            
            # Get the mode for categorical features
            categorical_cols = data.select_dtypes(exclude=[np.number]).columns
            cluster_modes = cluster_data[categorical_cols].mode().iloc[0]
            
            # Combine the characteristics
            characteristics = pd.concat([cluster_means, cluster_modes])
            
            cluster_characteristics[cluster] = characteristics
            
        return cluster_characteristics
    
    def explain_clusters(self, data):
        """
        Use SHAP to explain the clusters.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Lead data.
            
        Returns:
        --------
        dict
            SHAP explanations for each cluster.
        """
        # Preprocess the data
        data_scaled, feature_names = self.preprocess_data(data)
        
        # Get cluster assignments
        clusters = self.kmeans.predict(data_scaled)
        
        # Create a DataFrame with scaled data
        data_scaled_df = pd.DataFrame(data_scaled, columns=feature_names)
        
        # Add cluster assignments
        data_scaled_df['Cluster'] = clusters
        
        # Initialize dictionary for explanations
        explanations = {}
        
        # For each cluster, train a simple model to predict if a lead belongs to that cluster
        for cluster in range(self.n_clusters):
            # Create binary target: 1 if lead belongs to this cluster, 0 otherwise
            y = (data_scaled_df['Cluster'] == cluster).astype(int)
            
            # Train a simple model (e.g., Random Forest) to predict cluster membership
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            model.fit(data_scaled_df.drop('Cluster', axis=1), y)
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data_scaled_df.drop('Cluster', axis=1))
            
            # Store explanations
            explanations[cluster] = {
                'explainer': explainer,
                'shap_values': shap_values
            }
        
        return explanations
    
    def save_model(self, filepath):
        """
        Save the model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model.
        """
        joblib.dump({
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'pca': self.pca,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'feature_names': self.feature_names
        }, filepath)
        
    @classmethod
    def load_model(cls, filepath):
        """
        Load a model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model.
            
        Returns:
        --------
        LeadSegmentationModel
            Loaded model.
        """
        model_data = joblib.load(filepath)
        
        # Create a new instance
        model = cls(n_clusters=model_data['n_clusters'], random_state=model_data['random_state'])
        
        # Restore the model attributes
        model.kmeans = model_data['kmeans']
        model.scaler = model_data['scaler']
        model.pca = model_data['pca']
        model.feature_names = model_data['feature_names']
        
        return model

# Example usage
if __name__ == "__main__":
    # Sample data (would be replaced with actual lead data)
    data = {
        'time_since_last_interaction': np.random.randint(1, 365, 1000),
        'previous_page_views': np.random.randint(1, 50, 1000),
        'email_open_rate': np.random.random(1000),
        'interest_area': np.random.choice(['data_science', 'web_dev', 'mobile_dev', 'cybersecurity'], 1000),
        'career_stage': np.random.choice(['student', 'early_career', 'mid_career', 'senior'], 1000),
        'previously_viewed_courses': np.random.randint(0, 10, 1000),
        'quiz_attempts': np.random.randint(0, 5, 1000),
        'geographic_region': np.random.choice(['north', 'south', 'east', 'west'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    # Initialize and fit the model
    model = LeadSegmentationModel()
    model.fit(df)
    
    # Make predictions
    clusters = model.predict(df)
    print(f"Cluster distribution: {np.bincount(clusters)}")
    
    # Visualize clusters
    fig = model.visualize_clusters(df)
    
    # Get cluster characteristics
    characteristics = model.get_cluster_characteristics(df)
    for cluster, chars in characteristics.items():
        print(f"\nCluster {cluster} characteristics:")
        print(chars)
    
    # Save the model
    model.save_model('lead_segmentation_model.joblib')
