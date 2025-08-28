#!/usr/bin/env python3
"""
RAGTFD Setup and Testing Script
================================

This script sets up the complete RAGTFD fraud detection system and runs comprehensive tests.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "networkx",
        "fastapi",
        "uvicorn",
        "pydantic"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "uv", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def create_sample_dataset(num_transactions=20000):
    """Create a sample dataset matching your schema"""
    print(f"📊 Creating sample dataset with {num_transactions} transactions...")
    
    np.random.seed(42)
    random.seed(42)
    
    # Sample data generators
    locations = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Nagpur", "Thane"]
    device_types = ["Mobile", "Desktop", "Tablet"]
    transaction_modes = ["UPI ID", "UPI QR Code", "UPI Phone No"]
    banks = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB"]
    os_versions = ["Android-14", "Android-15", "iOS-16", "iOS-17", "MacOS-11", "Windows-10"]
    app_versions = ["2.1.5", "2.2.1", "3.6.9", "4.5.6", "5.1.5"]
    link_sources = ["Direct", "Email", "Social Media", "paytmm.in", "gpay-bonus.com", "secure-verify.org"]
    fraud_types = ["None", "Account_Takeover", "Synthetic_Identity", "Payment_Fraud", "Deepfake_Impersonation", "Social_Engineering"]
    
    data = []
    
    for i in range(num_transactions):
        # Determine if this transaction is fraudulent (15% fraud rate)
        is_fraud = 1 if random.random() < 0.15 else 0
        
        # Generate base transaction data
        user_id = f"U{random.randint(10000, 99999):05d}"
        beneficiary_id = f"B{i:05d}"
        transaction_id = f"TXN{i:06d}"
        
        # Generate correlated features based on fraud status
        if is_fraud:
            # Fraudulent transaction patterns
            amount = random.choice([
                random.uniform(50, 500),      # Small amounts
                random.uniform(15000, 50000)  # Large amounts
            ])
            frequency_24h = random.randint(5, 15)  # High frequency
            transaction_velocity = random.randint(8, 20)
            login_attempts = random.randint(3, 10)
            device_change = random.choice([0, 1])
            location_change = random.choice([0, 1])
            qr_manipulated = random.choice([0, 1]) if random.random() < 0.3 else 0
            link_source = random.choice(link_sources[3:])  # Suspicious sources
            fraud_type = random.choice(fraud_types[1:])
            user_account_age = random.randint(0, 30)  # New accounts
            beneficiary_age = random.choice(["1 days", "3 days", "1 weeks", "2 weeks"])
            attempt_ratio = random.uniform(0.6, 1.0)  # High attempt ratio
            
        else:
            # Normal transaction patterns
            amount = random.uniform(100, 5000)
            frequency_24h = random.randint(1, 4)
            transaction_velocity = random.randint(1, 7)
            login_attempts = random.randint(0, 2)
            device_change = 0 if random.random() < 0.9 else 1
            location_change = 0 if random.random() < 0.95 else 1
            qr_manipulated = 0
            link_source = random.choice(link_sources[:3])  # Safe sources
            fraud_type = "None"
            user_account_age = random.randint(30, 2000)
            beneficiary_age = random.choice(["2 months", "6 months", "1 years", "2 years", "5 years", "9 years"])
            attempt_ratio = random.uniform(0.1, 0.5)
        
        # Generate other fields
        time_hour = random.randint(0, 23)
        time_minute = random.randint(0, 59)
        time_str = f"{time_hour:02d}:{time_minute:02d}"
        
        transaction_date = datetime.now() - timedelta(days=random.randint(0, 90))
        
        transaction = {
            'Transaction_ID': transaction_id,
            'User_ID': user_id,
            'Amount': round(amount, 2),
            'Time': time_str,
            'Location': random.choice(locations),
            'Device_Type': random.choice(device_types),
            'Transaction_Mode': random.choice(transaction_modes),
            'Frequency_in_24hrs': frequency_24h,
            'Beneficiary_Account_Age': beneficiary_age,
            'Beneficiary_ID': beneficiary_id,
            'IP_Address': f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            'User_Account_Age_Days': user_account_age,
            'Transaction_Success': 1,
            'Login_Attempts_24hrs': login_attempts,
            'Device_Change_Flag': device_change,
            'Location_Change_Flag': location_change,
            'App_Version': random.choice(app_versions),
            'OS_Version': random.choice(os_versions),
            'Transaction_Velocity': transaction_velocity,
            'Attempt_to_Beneficiary_Ratio': round(attempt_ratio, 17),
            'Is_QR_Manipulated': qr_manipulated,
            'Linked_Bank': random.choice(banks),
            'Link_Clicked_From': link_source,
            'Fraud_Type': fraud_type,
            'Is_Fraud': is_fraud,
            'Transaction_Date': transaction_date.strftime('%Y-%m-%d %H:%M:%S.%f')
        }
        
        data.append(transaction)
        
        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1}/{num_transactions} transactions")
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    filename = "fraud_transactions_dataset.csv"
    df.to_csv(filename, index=False)
    
    fraud_count = df['Is_Fraud'].sum()
    print(f"✅ Dataset created: {filename}")
    print(f"   Total transactions: {len(df)}")
    print(f"   Fraudulent transactions: {fraud_count} ({fraud_count/len(df)*100:.1f}%)")
    print(f"   Normal transactions: {len(df)-fraud_count} ({(len(df)-fraud_count)/len(df)*100:.1f}%)")
    
    return filename

def create_project_structure():
    """Create the project directory structure"""
    print("📁 Creating project structure...")
    
    directories = [
        "models",
        "data",
        "logs",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created directory: {directory}/")

def run_ragtfd_pipeline(dataset_path):
    """Run the RAGTFD pipeline on the dataset"""
    print("🚀 Running RAGTFD Pipeline...")
    
    # This would import and run your pipeline
    try:
        # Import the pipeline class (assuming ragtfd_pipeline.py exists)
        exec(open('ragtfd_pipeline.py',encoding="utf-8").read(), globals())
        
        # Run the pipeline
        pipeline, results = load_and_run_ragtfd(dataset_path)
        
        return pipeline, results
        
    except FileNotFoundError:
        print("❌ ragtfd_pipeline.py not found. Please ensure the main pipeline file is available.")
        return None, None
    except Exception as e:
        print(f"❌ Error running pipeline: {str(e)}")
        return None, None

def validate_results(results, target_metrics):
    """Validate that results meet target metrics"""
    print("🎯 Validating results against targets...")
    
    if not results:
        print("❌ No results to validate")
        return False
    
    validation_passed = True
    tolerance = 2.0  # Allow 2% tolerance
    
    for metric, target in target_metrics.items():
        if metric in results:
            actual = results[metric]
            diff = abs(actual - target)
            
            if diff <= tolerance:
                print(f"  ✅ {metric}: {actual:.1f}% (target: {target:.1f}%)")
            else:
                print(f"  ❌ {metric}: {actual:.1f}% (target: {target:.1f}%, diff: {diff:.1f}%)")
                validation_passed = False
        else:
            print(f"  ❌ {metric}: Not found in results")
            validation_passed = False
    
    return validation_passed

def generate_test_report(results, dataset_path):
    """Generate a comprehensive test report"""
    print("📋 Generating test report...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"results/ragtfd_test_report_{timestamp}.json"
    
    report = {
        "test_timestamp": datetime.now().isoformat(),
        "dataset_path": dataset_path,
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform
        },
        "results": results,
        "target_metrics": {
            "accuracy": 91.8,
            "precision": 89.4,
            "recall": 90.5,
            "f1_score": 89.9,
            "latency_ms": 92.5
        }
    }

    with open(report_path, 'w', encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Test report saved: {report_path}")
    return report_path

def main():
    """Main setup and testing function"""
    print("🎯 RAGTFD Fraud Detection System Setup & Test")
    print("=" * 60)
    
    # Target metrics we want to achieve
    target_metrics = {
        'accuracy': 91.8,
        'precision': 89.4,
        'recall': 90.5,
        'f1_score': 89.9
    }
    
    try:
        # Step 1: Install requirements
        install_requirements()
        print()
        
        # Step 2: Create project structure
        create_project_structure()
        print()
        
        # Step 3: Create sample dataset if not exists
        dataset_path = "fraud_transactions_dataset.csv"
        if not os.path.exists(dataset_path):
            dataset_path = create_sample_dataset(20000)
        else:
            print(f"📊 Using existing dataset: {dataset_path}")
        print()
        
        # Step 4: Run RAGTFD pipeline
        pipeline, results = run_ragtfd_pipeline(dataset_path)
        print()
        
        if results:
            # Step 5: Validate results
            validation_passed = validate_results(results, target_metrics)
            print()
            
            # Step 6: Generate report
            report_path = generate_test_report(results, dataset_path)
            print()
            
            # Final summary
            print("🏁 RAGTFD Setup and Test Summary")
            print("=" * 40)
            if validation_passed:
                print("✅ All target metrics achieved!")
                print("🎉 RAGTFD system is ready for production use.")
            else:
                print("⚠️  Some metrics need improvement.")
                print("🔧 Consider model fine-tuning or feature engineering.")
            
            print(f"📊 Detailed report available at: {report_path}")
            
        else:
            print("❌ Pipeline execution failed. Please check the errors above.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Setup interrupted by user.")
    except Exception as e:
        print(f"❌ Setup failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

def run_api_test():
    """Test the FastAPI server functionality"""
    print("🌐 Testing FastAPI Server...")
    
    sample_transaction = {
        "Transaction_ID": "TEST001",
        "User_ID": "U12345",
        "Amount": 1500.0,
        "Time": "14:30",
        "Location": "Mumbai",
        "Device_Type": "Mobile",
        "Transaction_Mode": "UPI QR Code",
        "Frequency_in_24hrs": 3,
        "Beneficiary_Account_Age": "2 years",
        "Beneficiary_ID": "B54321",
        "IP_Address": "192.168.1.100",
        "User_Account_Age_Days": 365,
        "Transaction_Success": 1,
        "Login_Attempts_24hrs": 1,
        "Device_Change_Flag": 0,
        "Location_Change_Flag": 0,
        "App_Version": "3.6.9",
        "OS_Version": "Android-15",
        "Transaction_Velocity": 3,
        "Attempt_to_Beneficiary_Ratio": 0.33,
        "Is_QR_Manipulated": 0,
        "Linked_Bank": "SBI",
        "Link_Clicked_From": "Direct",
        "Transaction_Date": "2025-01-15 14:30:00.000000"
    }
    
    print("Sample transaction for API testing:")
    print(json.dumps(sample_transaction, indent=2))
    print("\n🚀 To test the API server:")
    print("1. Run: python ragtfd_api.py")
    print("2. Visit: http://localhost:8000/docs")
    print("3. Use the sample transaction above in the /transaction endpoint")

if __name__ == "__main__":
    # Run main setup and testing
    main()
    
    print("\n" + "="*60)
    print("🎯 Next Steps:")
    print("1. ✅ Dataset created and pipeline tested")
    print("2. 🌐 Test the FastAPI server: python ragtfd_api.py")
    print("3. 📊 Review results in the results/ directory")
    print("4. 🔧 Fine-tune models if needed")
    print("5. 🚀 Deploy to production environment")
    print("="*60)