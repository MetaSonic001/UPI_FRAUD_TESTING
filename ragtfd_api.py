from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import json
import time
from datetime import datetime
import uvicorn
import os
# Import the RAGTFD pipeline (assuming it's in ragtfd_pipeline.py)
# from ragtfd_pipeline import RAGTFDPipeline

class TransactionRequest(BaseModel):
    Transaction_ID: str
    User_ID: str
    Amount: float
    Time: str
    Location: str
    Device_Type: str
    Transaction_Mode: str
    Frequency_in_24hrs: int
    Beneficiary_Account_Age: str
    Beneficiary_ID: str
    IP_Address: str
    User_Account_Age_Days: int
    Transaction_Success: int
    Login_Attempts_24hrs: int
    Device_Change_Flag: int
    Location_Change_Flag: int
    App_Version: str
    OS_Version: str
    Transaction_Velocity: float
    Attempt_to_Beneficiary_Ratio: float
    Is_QR_Manipulated: int
    Linked_Bank: str
    Link_Clicked_From: str
    Fraud_Type: Optional[str] = "None"
    Transaction_Date: str

class FraudResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    decision: str  # "Approved", "Suspicious", "Blocked"
    processing_time_ms: float
    explanations: List[str]
    worker_scores: Dict[str, float]

app = FastAPI(title="RAGTFD Fraud Detection API", version="1.0.0")

# Global pipeline instance (will be initialized on startup)
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAGTFD pipeline on startup"""
    global pipeline
    print("🚀 Initializing RAGTFD Pipeline...")
    # Uncomment when you have the pipeline ready
    # pipeline = RAGTFDPipeline()
    # If you have a pre-trained model, load it here
    print("✅ RAGTFD Pipeline initialized successfully!")

@app.get("/")
async def root():
    return {"message": "RAGTFD Fraud Detection System", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/transaction", response_model=FraudResponse)
async def process_transaction(transaction: TransactionRequest):
    """Process a single transaction for fraud detection"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    start_time = time.time()
    
    try:
        # Convert Pydantic model to dict for processing
        transaction_dict = transaction.dict()
        
        # Process through RAGTFD pipeline
        features, processing_time = await pipeline.process_transaction(transaction_dict)
        
        # Make prediction (if model is trained)
        if pipeline.model is not None:
            # Convert features to format expected by model
            feature_array = [list(features.values())]
            predictions, probabilities = pipeline.predict(feature_array)
            
            fraud_probability = float(probabilities[0])
            is_fraud = bool(predictions[0])
        else:
            # Fallback: use simple rules
            fraud_probability = calculate_rule_based_score(features)
            is_fraud = fraud_probability > 0.5
        
        # Determine decision
        if fraud_probability > 0.8:
            decision = "Blocked"
        elif fraud_probability > 0.4:
            decision = "Suspicious"
        else:
            decision = "Approved"
        
        # Get explanations
        explanations = pipeline.explainability_worker.explain_decision(
            features, is_fraud, fraud_probability
        )
        
        # Calculate worker scores
        worker_scores = {
            "graph_score": calculate_graph_score(features),
            "temporal_score": calculate_temporal_score(features),
            "content_score": calculate_content_score(features),
            "overall_score": fraud_probability
        }
        
        total_processing_time = (time.time() - start_time) * 1000
        
        return FraudResponse(
            transaction_id=transaction.Transaction_ID,
            fraud_probability=fraud_probability,
            is_fraud=is_fraud,
            decision=decision,
            processing_time_ms=total_processing_time,
            explanations=explanations,
            worker_scores=worker_scores
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing transaction: {str(e)}")

@app.post("/batch_process")
async def batch_process_transactions(transactions: List[TransactionRequest]):
    """Process multiple transactions in batch"""
    results = []
    
    # Process transactions concurrently
    tasks = [process_transaction(txn) for txn in transactions]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            results.append({
                "transaction_id": transactions[i].Transaction_ID,
                "error": str(response)
            })
        else:
            results.append(response.dict())
    
    return {"processed_count": len(transactions), "results": results}

@app.get("/stats")
async def get_system_stats():
    """Get system performance statistics"""
    # This would track actual usage stats in production
    return {
        "total_transactions_processed": 0,
        "average_processing_time_ms": 0,
        "fraud_detection_rate": 0,
        "system_uptime": "0h 0m 0s"
    }

def calculate_rule_based_score(features: Dict) -> float:
    """Fallback rule-based fraud scoring when ML model is not available"""
    score = 0.0
    
    # Velocity-based scoring
    if features.get('velocity_anomaly', 0) == 1:
        score += 0.3
    
    if features.get('velocity_2min', 0) > 5:
        score += 0.2
    
    # Graph-based scoring
    if features.get('beneficiary_expansion', 0) > 3:
        score += 0.2
    
    if features.get('circular_flow', 0) == 1:
        score += 0.3
    
    # Content-based scoring
    if features.get('suspicious_link', 0) == 1:
        score += 0.25
    
    if features.get('qr_manipulated', 0) == 1:
        score += 0.4
    
    # Temporal scoring
    if features.get('is_night_transaction', 0) == 1 and features.get('amount', 0) > 10000:
        score += 0.15
    
    # Account age scoring
    if features.get('user_account_age', 0) < 30:  # New accounts
        score += 0.1
    
    if features.get('beneficiary_account_age_numeric', 0) < 7:  # Very new beneficiary
        score += 0.15
    
    return min(score, 1.0)  # Cap at 1.0

def calculate_graph_score(features: Dict) -> float:
    """Calculate graph worker specific score"""
    score = 0.0
    
    if features.get('beneficiary_expansion', 0) > 0:
        score += 0.3
    
    if features.get('circular_flow', 0) == 1:
        score += 0.4
    
    if features.get('device_diversity', 1) > 3:
        score += 0.2
    
    if features.get('user_degree_centrality', 0) > 10:
        score += 0.1
    
    return min(score, 1.0)

def calculate_temporal_score(features: Dict) -> float:
    """Calculate temporal worker specific score"""
    score = 0.0
    
    if features.get('velocity_anomaly', 0) == 1:
        score += 0.5
    
    if features.get('frequency_24h', 0) > 10:
        score += 0.2
    
    if features.get('is_night_transaction', 0) == 1:
        score += 0.1
    
    if features.get('amount_variance', 0) > 1000000:  # High variance in amounts
        score += 0.2
    
    return min(score, 1.0)

def calculate_content_score(features: Dict) -> float:
    """Calculate content worker specific score"""
    score = 0.0
    
    if features.get('suspicious_link', 0) == 1:
        score += 0.4
    
    if features.get('qr_manipulated', 0) == 1:
        score += 0.5
    
    if features.get('domain_reputation', 10) < 5:
        score += 0.2
    
    return min(score, 1.0)

if __name__ == "__main__":
    print("🚀 Starting RAGTFD FastAPI Server...")
    uvicorn.run(
        "ragtfd_api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
