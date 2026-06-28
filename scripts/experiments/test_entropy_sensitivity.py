import os
import sys
import pandas as pd
import numpy as np

# Thêm thư mục gốc vào PYTHONPATH
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_dir)

from backend.app.core.profiler.orchestrator import run_profiling
from backend.app.core.profiler.column_profile import DataRole

def test_entropy_sensitivity():
    print("🔬 Đang kiểm tra độ nhạy của ngưỡng Entropy (0.8)...")
    
    # Tạo dataset với các phân phối khác nhau
    # 1. target_balanced: 50/50 (Entropy ~ 1.0) -> Vượt ngưỡng 0.8
    # 2. target_imbalanced: 90/10 (Entropy ~ 0.47) -> Dưới ngưỡng 0.8
    # 3. target_edge: 70/30 (Entropy ~ 0.88) -> Vừa chớm vượt ngưỡng 0.8
    
    n = 1000
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "target_balanced": [0] * (n//2) + [1] * (n//2),
        "target_imbalanced": [0] * int(n*0.9) + [1] * int(n*0.1),
        "target_edge": [0] * int(n*0.7) + [1] * int(n*0.3)
    })
    
    profiles, suggested_target = run_profiling(df)
    
    print("\n📊 Kết quả phân tích Entropy từng cột:")
    for p in profiles:
        if "target" in p.name:
            print(f"Column: {p.name:20} | Entropy: {p.entropy:.4f} | Role: {p.inferred_role}")

    print(f"\n🎯 Hệ thống đề xuất Target: {suggested_target}")
    
    # Kiểm tra logic: target_imbalanced phải thắng vì nó có score = 1.0 (binary) + 0.5 (entropy < 0.8) = 1.5
    # Các cột khác chỉ có score = 1.0 (binary) + 0.0 (entropy > 0.8) = 1.0
    assert suggested_target == "target_imbalanced", f"Lẽ ra phải chọn target_imbalanced, nhưng lại chọn {suggested_target}"
    print("✅ Đã xác nhận: Hệ thống ưu tiên các cột có độ lệch (skew) tự nhiên của bài toán Churn.")

if __name__ == "__main__":
    test_entropy_sensitivity()
