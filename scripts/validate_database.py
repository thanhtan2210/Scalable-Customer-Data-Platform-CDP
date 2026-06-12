import os
import sys
from sqlalchemy import create_engine, text

# Tự động thêm thư mục gốc vào PYTHONPATH
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

try:
    from backend.app.core.config import DATABASE_URL
except ImportError as e:
    print(f"❌ Lỗi import config: {e}")
    sys.exit(1)

def test_database_foundation():
    print("🚀 Starting Database Foundation Validation (Phase 0)...")
    
    # 1. Tự động chuẩn hóa chuỗi URL
    url = DATABASE_URL.strip() # Xóa dấu cách thừa
    
    # Đảm bảo dùng driver psycopg2
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    
    # Hỗ trợ fix lỗi lỡ tay để dấu ngoặc vuông
    url = url.replace("[", "").replace("]", "")

    if "sqlite" in url:
        print("⚠️ DATABASE_URL đang dùng SQLite (local).")
    else:
        # In ra địa chỉ server để debug (ẩn pass)
        try:
            display_url = url.split("@")[-1]
            print(f"🔗 Đang thử kết nối tới: {display_url}")
        except:
            print("🔗 Đang thử kết nối tới Cloud DB...")

    try:
        # 2. Tạo engine với URL đã chuẩn hóa
        engine = create_engine(url, connect_args={"connect_timeout": 10})
        
        # 3. Thử query
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()
            print(f"✅ KẾT NỐI THÀNH CÔNG!")
            print(f"📌 Phiên bản DB: {version[0]}")
            
        print("\n🎯 DATABASE FOUNDATION IS READY!")
        
    except Exception as e:
        print(f"❌ Kết nối thất bại.")
        print(f"Chi tiết lỗi: {str(e)}")
        print("\n💡 KIỂM TRA LẠI FILE .env:")
        print("1. Đảm bảo mật khẩu KHÔNG nằm trong dấu [ ]")
        print("2. Nếu mật khẩu có ký tự @, hãy đổi nó thành %40")
        print("3. Đảm bảo dùng Port 6543 (Transaction Pooler)")

if __name__ == "__main__":
    test_database_foundation()
