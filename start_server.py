
import uvicorn
from ragtfd_api import app

if __name__ == "__main__":
    print("RAGTFD API Server starting...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True
    )
