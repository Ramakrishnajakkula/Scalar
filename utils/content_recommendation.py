# Content Recommendation Engine
# This module implements a hybrid recommendation system for educational content

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, List, Optional, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import pickle
import os

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class ContentRecommendationEngine:
    """
    Hybrid recommendation system that combines content-based filtering and collaborative filtering
    to recommend educational content to users.
    """
    
    def __init__(self, embedding_dimension: int = 32):
        """
        Initialize the recommendation engine.
        
        Parameters:
        -----------
        embedding_dimension : int
            Dimension of the embeddings for items and users.
        """
        self.embedding_dimension = embedding_dimension
        self.model = None
        self.user_model = None
        self.content_model = None
        self.content_data = None
        self.user_data = None
        self.interaction_data = None
        self.tfidf_vectorizer = None
        self.content_vectors = None
        self.content_mapping = None
        self.user_mapping = None
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text data for content-based filtering.
        
        Parameters:
        -----------
        text : str
            Text to preprocess.
            
        Returns:
        --------
        str
            Preprocessed text.
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        
        return ' '.join(tokens)
    
    def prepare_data(self, user_data: pd.DataFrame, content_data: pd.DataFrame, 
                    interaction_data: pd.DataFrame) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare data for training the recommendation model.
        
        Parameters:
        -----------
        user_data : pandas.DataFrame
            DataFrame containing user information.
        content_data : pandas.DataFrame
            DataFrame containing content information.
        interaction_data : pandas.DataFrame
            DataFrame containing user-content interactions.
            
        Returns:
        --------
        tuple
            Tuple containing train and test datasets.
        """
        # Store the original data
        self.user_data = user_data
        self.content_data = content_data
        self.interaction_data = interaction_data
        
        # Create user and content mappings (string IDs to integer indices)
        unique_user_ids = user_data['user_id'].unique()
        unique_content_ids = content_data['content_id'].unique()
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
        self.content_mapping = {content_id: idx for idx, content_id in enumerate(unique_content_ids)}
        
        # Create TF dataset from interaction data
        interactions = interaction_data.copy()
        interactions['user_idx'] = interactions['user_id'].map(self.user_mapping)
        interactions['content_idx'] = interactions['content_id'].map(self.content_mapping)
        
        # Convert to TF tensors
        user_ids = tf.convert_to_tensor(interactions['user_idx'].values, dtype=tf.int64)
        content_ids = tf.convert_to_tensor(interactions['content_idx'].values, dtype=tf.int64)
        ratings = tf.convert_to_tensor(interactions['rating'].values, dtype=tf.float32)
        
        # Create TF dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'user_id': user_ids,
            'content_id': content_ids,
            'rating': ratings
        })
        
        # Shuffle and split into train and test sets
        tf.random.set_seed(42)
        shuffled = dataset.shuffle(len(interactions), seed=42, reshuffle_each_iteration=False)
        
        train_size = int(0.8 * len(interactions))
        train_dataset = shuffled.take(train_size)
        test_dataset = shuffled.skip(train_size)
        
        # Cache and batch datasets
        train_dataset = train_dataset.batch(256).cache()
        test_dataset = test_dataset.batch(256).cache()
        
        # Prepare content-based filtering data
        self._prepare_content_based_data(content_data)
        
        return train_dataset, test_dataset
    
    def _prepare_content_based_data(self, content_data: pd.DataFrame) -> None:
        """
        Prepare data for content-based filtering.
        
        Parameters:
        -----------
        content_data : pandas.DataFrame
            DataFrame containing content information.
        """
        # Combine content fields into a single text field
        content_data['combined_text'] = content_data.apply(
            lambda row: ' '.join([
                str(row.get('title', '')), 
                str(row.get('description', '')),
                str(row.get('category', '')),
                str(row.get('tags', ''))
            ]), 
            axis=1
        )
        
        # Preprocess the combined text
        content_data['processed_text'] = content_data['combined_text'].apply(self.preprocess_text)
        
        # Create TF-IDF vectors
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.content_vectors = self.tfidf_vectorizer.fit_transform(content_data['processed_text'])
    
    def build_model(self, num_users: int, num_items: int) -> None:
        """
        Build the recommendation model.
        
        Parameters:
        -----------
        num_users : int
            Number of unique users.
        num_items : int
            Number of unique content items.
        """
        class HybridRecommendationModel(tfrs.models.Model):
            def __init__(self, num_users, num_items, embedding_dim):
                super().__init__()
                
                # User model
                self.user_model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=[1]),
                    tf.keras.layers.Embedding(num_users, embedding_dim),
                    tf.keras.layers.Dense(embedding_dim)
                ])
                
                # Content model
                self.content_model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=[1]),
                    tf.keras.layers.Embedding(num_items, embedding_dim),
                    tf.keras.layers.Dense(embedding_dim)
                ])
                
                # Task
                self.task = tfrs.tasks.Retrieval(
                    metrics=tfrs.metrics.FactorizedTopK(
                        candidates=tf.data.Dataset.from_tensor_slices(list(range(num_items)))
                        .map(lambda content_id: (content_id, self.content_model(tf.constant([content_id]))))
                    )
                )
            
            def compute_loss(self, features, training=False):
                user_embeddings = self.user_model(features["user_id"])
                content_embeddings = self.content_model(features["content_id"])
                
                return self.task(user_embeddings, content_embeddings, compute_metrics=not training)
        
        # Build the model
        self.model = HybridRecommendationModel(num_users=len(self.user_mapping), 
                                              num_items=len(self.content_mapping),
                                              embedding_dim=self.embedding_dimension)
        
        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
        
        # Store the user and content models separately for easier access
        self.user_model = self.model.user_model
        self.content_model = self.model.content_model
    
    def train(self, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, 
             num_epochs: int = 50) -> Dict:
        """
        Train the recommendation model.
        
        Parameters:
        -----------
        train_dataset : tf.data.Dataset
            Training dataset.
        test_dataset : tf.data.Dataset
            Testing dataset.
        num_epochs : int
            Number of training epochs.
            
        Returns:
        --------
        dict
            Training history.
        """
        # Build the model if it doesn't exist
        if self.model is None:
            self.build_model(len(self.user_mapping), len(self.content_mapping))
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=test_dataset,
            verbose=1
        )
        
        return history.history
    
    def get_content_based_recommendations(self, content_id: str, top_n: int = 5) -> List[Dict]:
        """
        Get content-based recommendations.
        
        Parameters:
        -----------
        content_id : str
            ID of the content to get recommendations for.
        top_n : int
            Number of recommendations to return.
            
        Returns:
        --------
        list
            List of recommended content items.
        """
        # Get the index of the content
        if content_id not in self.content_mapping:
            raise ValueError(f"Content ID {content_id} not found")
        
        content_idx = self.content_mapping[content_id]
        
        # Get the content vector
        content_vector = self.content_vectors[content_idx]
        
        # Compute similarity scores
        similarity_scores = cosine_similarity(content_vector, self.content_vectors)[0]
        
        # Get top N similar items (excluding the item itself)
        similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
        
        # Map back to content IDs
        reverse_mapping = {idx: content_id for content_id, idx in self.content_mapping.items()}
        recommendations = [
            {
                'content_id': reverse_mapping[idx],
                'similarity_score': float(similarity_scores[idx])
            }
            for idx in similar_indices
        ]
        
        return recommendations
    
    def get_collaborative_recommendations(self, user_id: str, top_n: int = 5) -> List[Dict]:
        """
        Get collaborative filtering recommendations.
        
        Parameters:
        -----------
        user_id : str
            ID of the user to get recommendations for.
        top_n : int
            Number of recommendations to return.
            
        Returns:
        --------
        list
            List of recommended content items.
        """
        # Check if the model is trained
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Check if user exists
        if user_id not in self.user_mapping:
            raise ValueError(f"User ID {user_id} not found")
        
        user_idx = self.user_mapping[user_id]
        
        # Create a TensorFlow index of content embeddings
        content_embeddings = tf.constant([
            self.content_model(tf.constant([i])).numpy().flatten()
            for i in range(len(self.content_mapping))
        ])
        
        # Get the user embedding
        user_embedding = self.user_model(tf.constant([user_idx])).numpy().flatten()
        
        # Compute similarity scores
        similarity_scores = tf.matmul(
            tf.expand_dims(tf.cast(user_embedding, tf.float32), 0),
            tf.cast(content_embeddings, tf.float32),
            transpose_b=True
        ).numpy().flatten()
        
        # Get top N items
        top_indices = similarity_scores.argsort()[::-1][:top_n]
        
        # Map back to content IDs
        reverse_mapping = {idx: content_id for content_id, idx in self.content_mapping.items()}
        recommendations = [
            {
                'content_id': reverse_mapping[idx],
                'score': float(similarity_scores[idx])
            }
            for idx in top_indices
        ]
        
        return recommendations
    
    def get_hybrid_recommendations(self, user_id: str, content_id: Optional[str] = None, 
                                  top_n: int = 5, weight_cf: float = 0.7) -> List[Dict]:
        """
        Get hybrid recommendations combining collaborative filtering and content-based filtering.
        
        Parameters:
        -----------
        user_id : str
            ID of the user to get recommendations for.
        content_id : str, optional
            ID of the content to get recommendations for (for content-based component).
        top_n : int
            Number of recommendations to return.
        weight_cf : float
            Weight for collaborative filtering (content-based weight = 1 - weight_cf).
            
        Returns:
        --------
        list
            List of recommended content items.
        """
        # Get collaborative filtering recommendations
        try:
            cf_recs = self.get_collaborative_recommendations(user_id, top_n=top_n*2)
        except ValueError:
            # If user not found, set empty recommendations and rely more on content-based
            cf_recs = []
            weight_cf = 0.0
        
        # Get content-based recommendations if content_id is provided
        if content_id is not None:
            try:
                cb_recs = self.get_content_based_recommendations(content_id, top_n=top_n*2)
            except ValueError:
                # If content not found, set empty recommendations and rely more on collaborative
                cb_recs = []
                weight_cf = 1.0
        else:
            # If no content_id is provided, rely solely on collaborative filtering
            cb_recs = []
            weight_cf = 1.0
        
        # If both methods failed, return empty list
        if not cf_recs and not cb_recs:
            return []
        
        # Combine recommendations
        combined_scores = {}
        
        # Process collaborative filtering recommendations
        for rec in cf_recs:
            content_id = rec['content_id']
            combined_scores[content_id] = combined_scores.get(content_id, 0) + rec['score'] * weight_cf
        
        # Process content-based recommendations
        for rec in cb_recs:
            content_id = rec['content_id']
            combined_scores[content_id] = combined_scores.get(content_id, 0) + rec['similarity_score'] * (1 - weight_cf)
        
        # Sort by combined score and get top_n
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Format results
        recommendations = [
            {
                'content_id': content_id,
                'score': float(score),
                'title': self.content_data.loc[self.content_data['content_id'] == content_id, 'title'].iloc[0],
                'category': self.content_data.loc[self.content_data['content_id'] == content_id, 'category'].iloc[0]
            }
            for content_id, score in sorted_recs
        ]
        
        return recommendations
    
    def get_recommendations_for_cold_user(self, user_features: Dict, top_n: int = 5) -> List[Dict]:
        """
        Get recommendations for a new user without interaction history.
        
        Parameters:
        -----------
        user_features : Dict
            Dictionary containing user features.
        top_n : int
            Number of recommendations to return.
            
        Returns:
        --------
        list
            List of recommended content items.
        """
        # Extract relevant features
        interest_area = user_features.get('interest_area', '')
        career_stage = user_features.get('career_stage', '')
        
        # Find similar users
        similar_users = []
        for idx, row in self.user_data.iterrows():
            similarity_score = 0
            
            # Check interest area similarity
            if interest_area and row.get('interest_area') == interest_area:
                similarity_score += 2
            
            # Check career stage similarity
            if career_stage and row.get('career_stage') == career_stage:
                similarity_score += 1
            
            if similarity_score > 0:
                similar_users.append({
                    'user_id': row['user_id'],
                    'similarity_score': similarity_score
                })
        
        # Sort by similarity score
        similar_users = sorted(similar_users, key=lambda x: x['similarity_score'], reverse=True)[:5]
        
        if not similar_users:
            # If no similar users found, fall back to popular content in the interest area
            if interest_area:
                content_by_interest = self.content_data[
                    self.content_data['category'].str.contains(interest_area, case=False, na=False)
                ]
                if len(content_by_interest) > 0:
                    # Get top content by popularity or rating
                    top_content = content_by_interest.sort_values('popularity', ascending=False).head(top_n)
                    
                    # Format results
                    recommendations = [
                        {
                            'content_id': row['content_id'],
                            'title': row['title'],
                            'category': row['category'],
                            'score': row['popularity']
                        }
                        for _, row in top_content.iterrows()
                    ]
                    
                    return recommendations
            
            # If no interest area or no matching content, return overall popular content
            top_content = self.content_data.sort_values('popularity', ascending=False).head(top_n)
            
            # Format results
            recommendations = [
                {
                    'content_id': row['content_id'],
                    'title': row['title'],
                    'category': row['category'],
                    'score': row['popularity']
                }
                for _, row in top_content.iterrows()
            ]
            
            return recommendations
        
        # Get recommendations for similar users
        all_recs = []
        for similar_user in similar_users:
            try:
                user_recs = self.get_collaborative_recommendations(similar_user['user_id'], top_n=top_n)
                for rec in user_recs:
                    rec['similarity_weight'] = similar_user['similarity_score']
                all_recs.extend(user_recs)
            except ValueError:
                continue
        
        # Combine and weight recommendations
        combined_scores = {}
        for rec in all_recs:
            content_id = rec['content_id']
            score = rec['score'] * rec['similarity_weight']
            
            if content_id in combined_scores:
                combined_scores[content_id] += score
            else:
                combined_scores[content_id] = score
        
        # Sort by combined score and get top_n
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Format results
        recommendations = [
            {
                'content_id': content_id,
                'score': float(score),
                'title': self.content_data.loc[self.content_data['content_id'] == content_id, 'title'].iloc[0],
                'category': self.content_data.loc[self.content_data['content_id'] == content_id, 'category'].iloc[0]
            }
            for content_id, score in sorted_recs
        ]
        
        return recommendations
    
    def save_model(self, directory: str) -> None:
        """
        Save the model and its components to disk.
        
        Parameters:
        -----------
        directory : str
            Directory to save the model.
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save TensorFlow model
        if self.model is not None:
            self.model.save_weights(os.path.join(directory, 'model_weights'))
        
        # Save TF-IDF vectorizer
        if self.tfidf_vectorizer is not None:
            with open(os.path.join(directory, 'tfidf_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        # Save mappings
        if self.user_mapping is not None:
            with open(os.path.join(directory, 'user_mapping.pkl'), 'wb') as f:
                pickle.dump(self.user_mapping, f)
        
        if self.content_mapping is not None:
            with open(os.path.join(directory, 'content_mapping.pkl'), 'wb') as f:
                pickle.dump(self.content_mapping, f)
        
        # Save content vectors
        if self.content_vectors is not None:
            with open(os.path.join(directory, 'content_vectors.pkl'), 'wb') as f:
                pickle.dump(self.content_vectors, f)
    
    @classmethod
    def load_model(cls, directory: str, user_data: pd.DataFrame, 
                  content_data: pd.DataFrame, interaction_data: pd.DataFrame) -> 'ContentRecommendationEngine':
        """
        Load a model from disk.
        
        Parameters:
        -----------
        directory : str
            Directory where the model is saved.
        user_data : pandas.DataFrame
            DataFrame containing user information.
        content_data : pandas.DataFrame
            DataFrame containing content information.
        interaction_data : pandas.DataFrame
            DataFrame containing user-content interactions.
            
        Returns:
        --------
        ContentRecommendationEngine
            Loaded model.
        """
        # Create a new instance
        engine = cls()
        
        # Set the data
        engine.user_data = user_data
        engine.content_data = content_data
        engine.interaction_data = interaction_data
        
        # Load mappings
        with open(os.path.join(directory, 'user_mapping.pkl'), 'rb') as f:
            engine.user_mapping = pickle.load(f)
        
        with open(os.path.join(directory, 'content_mapping.pkl'), 'rb') as f:
            engine.content_mapping = pickle.load(f)
        
        # Load TF-IDF vectorizer
        with open(os.path.join(directory, 'tfidf_vectorizer.pkl'), 'rb') as f:
            engine.tfidf_vectorizer = pickle.load(f)
        
        # Load content vectors
        with open(os.path.join(directory, 'content_vectors.pkl'), 'rb') as f:
            engine.content_vectors = pickle.load(f)
        
        # Build the model
        engine.build_model(len(engine.user_mapping), len(engine.content_mapping))
        
        # Load weights
        engine.model.load_weights(os.path.join(directory, 'model_weights'))
        
        # Store the user and content models separately for easier access
        engine.user_model = engine.model.user_model
        engine.content_model = engine.model.content_model
        
        return engine

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    # Sample user data
    user_data = pd.DataFrame({
        'user_id': [f"user_{i}" for i in range(100)],
        'interest_area': np.random.choice(['data_science', 'web_dev', 'mobile_dev', 'cybersecurity'], 100),
        'career_stage': np.random.choice(['student', 'early_career', 'mid_career', 'senior'], 100)
    })
    
    # Sample content data
    content_data = pd.DataFrame({
        'content_id': [f"content_{i}" for i in range(50)],
        'title': [f"Course {i}" for i in range(50)],
        'description': [f"Description for course {i}" for i in range(50)],
        'category': np.random.choice(['data_science', 'web_dev', 'mobile_dev', 'cybersecurity'], 50),
        'tags': [f"tag1,tag2,tag{i}" for i in range(50)],
        'popularity': np.random.rand(50) * 10
    })
    
    # Sample interaction data
    interactions = []
    for user_id in user_data['user_id']:
        # Each user interacts with some random content
        for _ in range(np.random.randint(5, 15)):
            content_id = np.random.choice(content_data['content_id'])
            rating = np.random.rand() * 5  # Rating between 0 and 5
            interactions.append({
                'user_id': user_id,
                'content_id': content_id,
                'rating': rating
            })
    
    interaction_data = pd.DataFrame(interactions)
    
    # Create and train the recommendation engine
    engine = ContentRecommendationEngine(embedding_dimension=32)
    train_dataset, test_dataset = engine.prepare_data(user_data, content_data, interaction_data)
    engine.build_model(len(engine.user_mapping), len(engine.content_mapping))
    history = engine.train(train_dataset, test_dataset, num_epochs=5)  # Reduced epochs for example
    
    # Test different recommendation methods
    user_id = 'user_0'
    content_id = 'content_0'
    
    print("Collaborative filtering recommendations:")
    cf_recs = engine.get_collaborative_recommendations(user_id, top_n=3)
    for rec in cf_recs:
        print(f"Content: {rec['content_id']}, Score: {rec['score']:.4f}")
    
    print("\nContent-based recommendations:")
    cb_recs = engine.get_content_based_recommendations(content_id, top_n=3)
    for rec in cb_recs:
        print(f"Content: {rec['content_id']}, Similarity: {rec['similarity_score']:.4f}")
    
    print("\nHybrid recommendations:")
    hybrid_recs = engine.get_hybrid_recommendations(user_id, content_id, top_n=3)
    for rec in hybrid_recs:
        print(f"Content: {rec['content_id']}, Title: {rec['title']}, Score: {rec['score']:.4f}")
    
    print("\nRecommendations for cold user:")
    cold_user = {'interest_area': 'data_science', 'career_stage': 'student'}
    cold_recs = engine.get_recommendations_for_cold_user(cold_user, top_n=3)
    for rec in cold_recs:
        print(f"Content: {rec['content_id']}, Title: {rec['title']}, Score: {rec['score']:.4f}")
