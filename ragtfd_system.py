import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import time
import asyncio
from datetime import datetime, timedelta
import re
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

class GraphWorker:
    """Graph-based fraud detection using NetworkX"""
    
    def __init__(self):
        self.user_graph = nx.Graph()
        self.device_graph = nx.Graph()
        self.beneficiary_graph = nx.Graph()
        self.user_histories = {}
        
    def extract_graph_features(self, transaction):
        """Extract graph-based features for a transaction"""
        user_id = transaction['User_ID']
        beneficiary_id = transaction['Beneficiary_ID']
        device_type = transaction['Device_Type']
        ip_address = transaction['IP_Address']
        
        features = {}
        
        # User-Beneficiary Graph Features
        if not self.user_graph.has_edge(user_id, beneficiary_id):
            self.user_graph.add_edge(user_id, beneficiary_id)
        
        # Calculate centrality metrics
        try:
            user_degree = self.user_graph.degree(user_id)
            features['user_degree_centrality'] = user_degree
            
            # Detect sudden beneficiary expansion
            if user_id not in self.user_histories:
                self.user_histories[user_id] = {'beneficiaries': set(), 'devices': set()}
            
            prev_beneficiaries = len(self.user_histories[user_id]['beneficiaries'])
            self.user_histories[user_id]['beneficiaries'].add(beneficiary_id)
            current_beneficiaries = len(self.user_histories[user_id]['beneficiaries'])
            
            features['beneficiary_expansion'] = current_beneficiaries - prev_beneficiaries
            features['total_beneficiaries'] = current_beneficiaries
            
            # Device switching detection
            self.user_histories[user_id]['devices'].add(device_type)
            features['device_diversity'] = len(self.user_histories[user_id]['devices'])
            
        except:
            features['user_degree_centrality'] = 0
            features['beneficiary_expansion'] = 0
            features['total_beneficiaries'] = 1
            features['device_diversity'] = 1
        
        # Check for circular flows (simplified)
        try:
            if nx.has_path(self.user_graph, beneficiary_id, user_id):
                features['circular_flow'] = 1
            else:
                features['circular_flow'] = 0
        except:
            features['circular_flow'] = 0
            
        return features

class TemporalWorker:
    """Temporal pattern analysis for velocity and timing anomalies"""
    
    def __init__(self):
        self.user_transactions = {}
        self.velocity_threshold = 5  # transactions per 2 minutes
        
    def extract_temporal_features(self, transaction):
        """Extract temporal features focusing on velocity and timing patterns"""
        user_id = transaction['User_ID']
        transaction_time = transaction['Time']
        amount = float(transaction['Amount'])
        frequency_24h = int(transaction['Frequency_in_24hrs'])
        velocity = float(transaction['Transaction_Velocity'])
        
        features = {}
        
        # Initialize user history
        if user_id not in self.user_transactions:
            self.user_transactions[user_id] = []
            
        current_time = datetime.strptime(f"2025-01-01 {transaction_time}", "%Y-%m-%d %H:%M")
        
        # Add current transaction to history
        self.user_transactions[user_id].append({
            'time': current_time,
            'amount': amount
        })
        
        # Keep only last 24 hours of transactions
        cutoff_time = current_time - timedelta(hours=24)
        self.user_transactions[user_id] = [
            tx for tx in self.user_transactions[user_id] 
            if tx['time'] >= cutoff_time
        ]
        
        # Calculate velocity features
        recent_transactions = [
            tx for tx in self.user_transactions[user_id]
            if (current_time - tx['time']).total_seconds() <= 120  # 2 minutes
        ]
        
        features['velocity_2min'] = len(recent_transactions)
        features['velocity_anomaly'] = 1 if len(recent_transactions) > self.velocity_threshold else 0
        
        # Transaction frequency analysis
        features['frequency_24h'] = frequency_24h
        features['transaction_velocity'] = velocity
        
        # Amount-based velocity
        if len(recent_transactions) > 1:
            recent_amounts = [tx['amount'] for tx in recent_transactions]
            features['avg_amount_2min'] = np.mean(recent_amounts)
            features['amount_variance'] = np.var(recent_amounts)
        else:
            features['avg_amount_2min'] = amount
            features['amount_variance'] = 0
            
        # Time-based patterns
        hour = current_time.hour
        features['is_night_transaction'] = 1 if hour < 6 or hour > 22 else 0
        features['is_business_hours'] = 1 if 9 <= hour <= 17 else 0
        
        return features

class ContentWorker:
    """Content analysis for URLs, links, and transaction modes"""
    
    def __init__(self):
        self.suspicious_domains = [
            'paytmm.in', 'gpay-bonus.com', 'phonepe-offer.net', 
            'secure-verify.org', 'upi-reward.co', 'bank-secure.net'
        ]
        self.legitimate_domains = [
            'paytm.in', 'gpay.com', 'phonepe.com', 
            'direct', 'email', 'social media'
        ]
        
    def extract_content_features(self, transaction):
        """Extract content-based features from URLs and transaction modes"""
        link_source = str(transaction['Link_Clicked_From']).lower()
        transaction_mode = str(transaction['Transaction_Mode']).lower()
        is_qr_manipulated = int(transaction['Is_QR_Manipulated'])
        
        features = {}
        
        # QR manipulation detection
        features['qr_manipulated'] = is_qr_manipulated
        
        # URL/Link analysis
        features['suspicious_link'] = self._analyze_link_suspicion(link_source)
        features['link_legitimacy_score'] = self._calculate_link_score(link_source)
        
        # Transaction mode analysis
        features['is_qr_transaction'] = 1 if 'qr' in transaction_mode else 0
        features['is_phone_transaction'] = 1 if 'phone' in transaction_mode else 0
        features['is_id_transaction'] = 1 if 'id' in transaction_mode else 0
        
        # Domain reputation
        features['domain_reputation'] = self._get_domain_reputation(link_source)
        
        return features
    
    def _analyze_link_suspicion(self, link_source):
        """Analyze if a link source is suspicious"""
        if link_source in ['direct', 'email', 'social media']:
            return 0
            
        # Check for typosquatting
        for suspicious in self.suspicious_domains:
            if suspicious in link_source:
                return 1
                
        # Check for suspicious patterns
        suspicious_patterns = [
            r'[0-9]+\.tk$', r'bit\.ly', r'tinyurl', r'secure-.*\..*',
            r'verify-.*\..*', r'bonus-.*\..*', r'offer-.*\..*'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, link_source):
                return 1
                
        return 0
    
    def _calculate_link_score(self, link_source):
        """Calculate legitimacy score for link"""
        if link_source in ['direct']:
            return 1.0
        elif link_source in ['email', 'social media']:
            return 0.8
        elif any(domain in link_source for domain in self.legitimate_domains):
            return 0.9
        elif any(domain in link_source for domain in self.suspicious_domains):
            return 0.1
        else:
            return 0.5
    
    def _get_domain_reputation(self, link_source):
        """Get domain reputation score"""
        if link_source in ['direct']:
            return 10
        elif link_source in ['email']:
            return 8
        elif link_source in ['social media']:
            return 7
        else:
            return 5

class ExplainabilityWorker:
    """Aggregates results and provides explanations"""
    
    def __init__(self):
        self.feature_importance = {}
        
    def aggregate_features(self, graph_features, temporal_features, content_features, transaction):
        """Combine all features and add additional derived features"""
        combined_features = {}
        
        # Add all worker features
        combined_features.update(graph_features)
        combined_features.update(temporal_features)
        combined_features.update(content_features)
        
        # Add original transaction features
        combined_features.update({
            'amount': float(transaction['Amount']),
            'user_account_age': int(transaction['User_Account_Age_Days']),
            'beneficiary_account_age_numeric': self._parse_account_age(transaction['Beneficiary_Account_Age']),
            'login_attempts': int(transaction['Login_Attempts_24hrs']),
            'device_change_flag': int(transaction['Device_Change_Flag']),
            'location_change_flag': int(transaction['Location_Change_Flag']),
            'transaction_success': int(transaction['Transaction_Success']),
            'attempt_to_beneficiary_ratio': float(transaction['Attempt_to_Beneficiary_Ratio'])
        })
        
        return combined_features
    
    def _parse_account_age(self, age_str):
        """Convert account age string to numeric days"""
        if 'days' in str(age_str).lower():
            return int(re.search(r'\d+', str(age_str)).group())
        elif 'years' in str(age_str).lower():
            return int(re.search(r'\d+', str(age_str)).group()) * 365
        elif 'months' in str(age_str).lower():
            return int(re.search(r'\d+', str(age_str)).group()) * 30
        else:
            return 0
    
    def explain_decision(self, features, prediction, probability):
        """Provide explanation for the fraud decision"""
        explanations = []
        
        if features.get('velocity_anomaly', 0) == 1:
            explanations.append(f"High velocity: {features.get('velocity_2min', 0)} transactions in 2 minutes")
            
        if features.get('suspicious_link', 0) == 1:
            explanations.append("Suspicious link detected")
            
        if features.get('qr_manipulated', 0) == 1:
            explanations.append("QR code manipulation detected")
            
        if features.get('beneficiary_expansion', 0) > 3:
            explanations.append(f"Sudden beneficiary expansion: {features.get('beneficiary_expansion', 0)} new beneficiaries")
            
        if features.get('circular_flow', 0) == 1:
            explanations.append("Circular transaction flow detected")
            
        if features.get('is_night_transaction', 0) == 1 and features.get('amount', 0) > 10000:
            explanations.append("High-value night transaction")
            
        return explanations

class RAGTFDPipeline:
    """Main RAGTFD fraud detection pipeline"""
    
    def __init__(self):
        self.graph_worker = GraphWorker()
        self.temporal_worker = TemporalWorker()
        self.content_worker = ContentWorker()
        self.explainability_worker = ExplainabilityWorker()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    async def process_transaction(self, transaction):
        """Process a single transaction through all workers"""
        start_time = time.time()
        
        # Run all workers in parallel
        graph_features = self.graph_worker.extract_graph_features(transaction)
        temporal_features = self.temporal_worker.extract_temporal_features(transaction)
        content_features = self.content_worker.extract_content_features(transaction)
        
        # Aggregate features
        combined_features = self.explainability_worker.aggregate_features(
            graph_features, temporal_features, content_features, transaction
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return combined_features, processing_time
    
    def prepare_training_data(self, df):
        """Prepare training data by processing all transactions"""
        print("Processing transactions for feature extraction...")
        features_list = []
        processing_times = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(df)} transactions")
                
            # Process transaction synchronously for training
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            features, proc_time = loop.run_until_complete(self.process_transaction(row))
            features_list.append(features)
            processing_times.append(proc_time)
            loop.close()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        return features_df, processing_times
    
    def train_model(self, X_train, y_train):
        """Train the fraud detection model"""
        print("Training fraud detection model...")
        
        # Scale numerical features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Use XGBoost for better performance
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train_scaled, y_train)
        print("Model training completed!")
        
    def predict(self, X_test):
        """Make predictions on test data"""
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
        return predictions, probabilities
    
    def evaluate_model(self, y_true, y_pred, processing_times):
        """Evaluate model performance"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        avg_latency = np.mean(processing_times)
        
        return {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'latency_ms': avg_latency
        }

def load_and_run_ragtfd(csv_file_path):
    """Main function to load CSV and run RAGTFD pipeline"""
    print("🚀 Starting RAGTFD Fraud Detection System")
    print("="*50)
    
    # Load data
    print("📊 Loading transaction data...")
    df = pd.read_csv(csv_file_path)
    print(f"Loaded {len(df)} transactions")
    
    # Initialize pipeline
    pipeline = RAGTFDPipeline()
    
    # Process all transactions to extract features
    features_df, processing_times = pipeline.prepare_training_data(df)
    
    # Prepare labels
    y = df['Is_Fraud'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    pipeline.train_model(X_train, y_train)
    
    # Get test processing times
    test_processing_times = processing_times[len(X_train):]
    
    # Make predictions
    y_pred, y_prob = pipeline.predict(X_test)
    
    # Evaluate model
    metrics = pipeline.evaluate_model(y_test, y_pred, test_processing_times)
    
    # Display results
    print("\n🎯 RAGTFD Performance Results")
    print("="*50)
    print(f"Accuracy:     {metrics['accuracy']:.1f}%")
    print(f"Precision:    {metrics['precision']:.1f}%")
    print(f"Recall:       {metrics['recall']:.1f}%")
    print(f"F1-Score:     {metrics['f1_score']:.1f}%")
    print(f"Latency:      {metrics['latency_ms']:.1f}ms")
    print("="*50)
    
    # Feature importance analysis
    if hasattr(pipeline.model, 'feature_importances_'):
        feature_importance = dict(zip(features_df.columns, pipeline.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\n🔍 Top 10 Most Important Features:")
        for feature, importance in top_features:
            print(f"{feature}: {importance:.4f}")
    
    return pipeline, metrics

# Example usage
if __name__ == "__main__":
    # Replace 'your_dataset.csv' with your actual CSV file path
    csv_file_path = "fraud_transactions.csv"
    
    try:
        pipeline, results = load_and_run_ragtfd(csv_file_path)
        print("\n✅ RAGTFD pipeline executed successfully!")
    except FileNotFoundError:
        print("❌ CSV file not found. Please ensure the file path is correct.")
    except Exception as e:
        print(f"❌ Error running pipeline: {str(e)}")
