"""
Test script to verify QR generation fix
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set minimal environment variables to avoid validation errors
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["REDIS_URL"] = "redis://localhost:6379"

def test_qr_generation_logic():
    """Test the QR generation logic without actually starting services"""
    print("🧪 Testing QR generation logic...")
    
    # Test event handler logic
    qr_received = False
    qr_code_data = None
    user_id = "test-user"
    
    def on_qr_received(qr_data):
        nonlocal qr_received, qr_code_data
        qr_received = True
        qr_code_data = qr_data.get('qr') if isinstance(qr_data, dict) else qr_data
        print(f"📱 QR Code REAL generado para usuario {user_id}: {qr_code_data[:50]}...")
    
    # Test with dict format (what WhatsApp Web.js sends)
    test_qr_dict = {"qr": "2@abc123def456ghi789,jkl012mno345,pqr678==", "timestamp": "2025-01-01T00:00:00Z"}
    on_qr_received(test_qr_dict)
    
