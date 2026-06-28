import os
import sys
from sqlalchemy.engine import make_url

# Tự động thêm thư mục gốc vào PYTHONPATH
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_dir)

def diagnose_url():
    print("🔎 Đang chẩn đoán cấu trúc DATABASE_URL...")
    
    try:
        from backend.app.core.config import DATABASE_URL
        url_str = DATABASE_URL.strip()
    except Exception as e:
        print(f"❌ Không đọc được config: {e}")
        return

    # 1. Kiểm tra tiền tố
    if not url_str.startswith("postgresql"):
        print("❌ Lỗi: URL phải bắt đầu bằng 'postgresql://'")
        return

    # 2. Thử Parse bằng thư viện chuẩn của SQLAlchemy
    try:
        # Tạm thời chuẩn hóa để parse thử
        if url_str.startswith("postgresql://") and "psycopg2" not in url_str:
            url_to_parse = url_str.replace("postgresql://", "postgresql+psycopg2://", 1)
        else:
            url_to_parse = url_str
            
        res = make_url(url_to_parse)
        
        print("✅ Cấu trúc URL hợp lệ!")
        print(f"--- Thông tin giải mã (Đã ẩn mật khẩu) ---")
        print(f"Driver:   {res.drivername}")
        print(f"Username: {res.username}")
        print(f"Host:     {res.host}")
        print(f"Port:     {res.port}")
        print(f"Database: {res.database}")
        print(f"------------------------------------------")
        print("👉 Nếu thông tin trên đúng, hãy chạy: python scripts/validation/validate_database.py")

    except Exception as e:
        print("❌ Lỗi Parse URL: Không thể phân tích cú pháp chuỗi này.")
        print(f"Chi tiết: {e}")
        print("\n💡 HƯỚNG DẪN FIX LỖI:")
        
        if "@" in url_str:
            parts = url_str.split("@")
            prefix_part = parts[0] # postgresql://user:pass
            if ":" not in prefix_part.replace("://", ""):
                 print("-> Thiếu dấu ':' giữa Username và Mật khẩu.")
            
            password = prefix_part.split(":")[-1]
            if any(c in password for c in ["#", "?", "/", "@"]):
                print(f"-> CẢNH BÁO: Mật khẩu có ký tự đặc biệt ({[c for c in ['#','?','/','@'] if c in password]}).")
                print("   Hãy vào Supabase đổi mật khẩu chỉ gồm Chữ và Số cho đơn giản, hoặc mã hóa URL các ký tự đó.")

if __name__ == "__main__":
    diagnose_url()
