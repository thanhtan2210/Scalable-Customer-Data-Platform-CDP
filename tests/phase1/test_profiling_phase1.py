import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError

from backend.app.core.profiler.column_profile import ColumnProfile, DataRole
from backend.app.core.profiler.layer1_stats import profile_column
from backend.app.core.profiler.layer2_semantic import detect_semantic
from backend.app.core.profiler.orchestrator import run_profiling, detect_target, check_leakage

def test_column_profile_schema():
    """
    1. ColumnProfile schema:
       - Import không lỗi
       - Khởi tạo với đủ fields không raise exception
       - Field potential_leakage default False
       - Field confidence_score từ chối giá trị > 1.0
    """
    # Khởi tạo hợp lệ
    profile = ColumnProfile(
        name="test_col",
        inferred_dtype="float64",
        inferred_role=DataRole.NUMERIC,
        confidence_score=0.8,
        null_pct=0.1,
        unique_count=100,
        entropy=2.5,
        transform_strategy="standard",
        impute_strategy="median"
    )
    assert profile.name == "test_col"
    assert profile.potential_leakage is False, "potential_leakage phải default là False!"
    
    # Từ chối confidence_score > 1.0
    with pytest.raises(ValidationError):
        ColumnProfile(
            name="invalid_col",
            inferred_dtype="float64",
            inferred_role=DataRole.NUMERIC,
            confidence_score=1.2, # > 1.0
            null_pct=0.1,
            unique_count=100,
            entropy=2.5
        )

def test_layer1_stats_no_hardcoding():
    """
    2. Layer 1 — không hardcode tên cột:
       - Chạy profile_column() với Series tên "xyz_random_col_123"
       - Expect: không crash, trả về dict có entropy, null_pct,
         cardinality_ratio, inferred_role
    """
    series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0] * 10, name="xyz_random_col_123")
    
    try:
        result = profile_column(series)
    except Exception as e:
        pytest.fail(f"profile_column crashed với tên cột ngẫu nhiên: {e}")
        
    assert result["name"] == "xyz_random_col_123"
    assert "entropy" in result, "Thiếu trường 'entropy' trong kết quả trả về của Layer 1!"
    assert "null_pct" in result, "Thiếu trường 'null_pct' trong kết quả trả về của Layer 1!"
    assert "inferred_role" in result, "Thiếu trường 'inferred_role' trong kết quả trả về của Layer 1!"
    assert "cardinality_ratio" in result, "Thiếu trường 'cardinality_ratio' trong kết quả trả về của Layer 1!"

def test_target_detection_domain_agnostic():
    """
    3. Target detection — domain agnostic:
       - Dataset không có từ khóa churn/target/label
       - Cột binary low-entropy phải được suggest làm target
       - Keyword match chỉ tăng confidence, không quyết định
    """
    # Dataset hoàn toàn không chứa từ khóa nhạy cảm
    df = pd.DataFrame({
        "customer_id": range(100),
        "feature_a": np.random.rand(100),
        "feature_b": np.random.rand(100),
        "outcome_bin": [0, 1] * 50  # Binary column, low-entropy
    })
    
    profiles = [
        {"name": "customer_id", "unique_count": 100, "entropy": 1.0, "inferred_role": DataRole.ID},
        {"name": "feature_a", "unique_count": 100, "entropy": 0.95, "inferred_role": DataRole.NUMERIC},
        {"name": "feature_b", "unique_count": 100, "entropy": 0.92, "inferred_role": DataRole.NUMERIC},
        {"name": "outcome_bin", "unique_count": 2, "entropy": 0.5, "inferred_role": DataRole.CATEGORICAL}
    ]
    
    suggested = detect_target(profiles, df)
    assert suggested.recommended_target == "outcome_bin", f"Không tự động nhận diện được outcome_bin làm target. Nhận diện nhầm: {suggested.recommended_target}"
    
    # Kiểm tra keyword match chỉ tăng confidence chứ không quyết định
    # Cột có keyword nhưng unique_count quá lớn không được chọn làm target
    df_keyword = pd.DataFrame({
        "some_churn_col": range(100), # Có keyword nhưng unique_count = 100, entropy cao
        "binary_col": [0, 1] * 50     # Không keyword nhưng unique_count = 2, entropy thấp
    })
    
    profiles_keyword = [
        {"name": "some_churn_col", "unique_count": 100, "entropy": 1.0, "inferred_role": DataRole.NUMERIC},
        {"name": "binary_col", "unique_count": 2, "entropy": 0.5, "inferred_role": DataRole.CATEGORICAL}
    ]
    
    suggested_keyword = detect_target(profiles_keyword, df_keyword)
    assert suggested_keyword.recommended_target == "binary_col", f"Bị đánh lừa bởi keyword match! Nhận diện: {suggested_keyword.recommended_target}"

def test_leakage_detection():
    """
    4. Leakage detection:
       - Cột copy 100% từ target → potential_leakage=True,
         leakage_score > 0.95
       - Hệ thống không tự drop cột đó
    """
    target = [0, 1] * 50
    df = pd.DataFrame({
        "target": target,
        "leakage_col": target,  # Copy 100%
        "feature": np.random.rand(100)
    })
    
    profiles = [
        {"name": "target", "inferred_dtype": "int64", "unique_count": 2, "inferred_role": DataRole.TARGET},
        {"name": "leakage_col", "inferred_dtype": "int64", "unique_count": 2, "inferred_role": DataRole.CATEGORICAL},
        {"name": "feature", "inferred_dtype": "float64", "unique_count": 100, "inferred_role": DataRole.NUMERIC}
    ]
    
    # Run leakage check
    check_leakage("target", profiles, df)
    
    leakage_profile = next(p for p in profiles if p["name"] == "leakage_col")
    assert leakage_profile.get("potential_leakage") is True, "Cột sao chép 100% phải được gắn cờ potential_leakage!"
    assert leakage_profile.get("leakage_score") is not None and leakage_profile["leakage_score"] > 0.95
    
    # Đảm bảo hệ thống không tự động lọc bỏ cột này trong run_profiling
    final_profiles, _ = run_profiling(df)
    assert any(p.name == "leakage_col" for p in final_profiles), "Hệ thống tự động xóa cột leakage là sai, phải giữ lại để người dùng tự xác nhận!"

def test_layer2_text_role():
    """
    5. Layer 2 — TEXT role:
       - Cột có mean_length > 50 → inferred_role = TEXT
       - transform_strategy = "tfidf", impute_strategy = "constant"
    """
    # Tạo dataset 2 cột: 1 cột target, 1 cột text_col chứa các câu văn độc nhất
    long_texts = [
        f"Đây là câu văn bản thứ {i} dùng để kiểm tra tính năng nhận diện dữ liệu văn bản tự do TEXT role."
        for i in range(50)
    ]
    df = pd.DataFrame({
        "text_col": long_texts,
        "target_col": [0, 1] * 25
    })
    
    final_profiles, _ = run_profiling(df)
    
    text_profile = next(p for p in final_profiles if p.name == "text_col")
    assert text_profile.inferred_role == DataRole.TEXT, f"Cột văn bản dài bị gán sai role: {text_profile.inferred_role}"
    assert text_profile.transform_strategy == "tfidf"
    assert text_profile.impute_strategy == "constant"

def test_orchestrator_output_contract():
    """
    6. Orchestrator output contract:
       - run_profiling(df) trả về tuple (List[ColumnProfile], str)
       - List chứa đủ tất cả cột kể cả ID và IGNORE
       - suggested_target là string tên cột, không phải ColumnProfile
    """
    df = pd.DataFrame({
        "id_col": range(100),
        "const_col": [1] * 100, # unique_count = 1 -> IGNORE
        "numeric_col": np.random.rand(100),
        "target_col": [0, 1] * 50
    })
    
    res = run_profiling(df)
    assert isinstance(res, tuple) and len(res) == 2, "run_profiling phải trả về một tuple gồm 2 phần tử!"
    
    from backend.app.core.profiler.target_analysis import TargetAnalysis
    profiles, suggested_target = res
    assert isinstance(profiles, list)
    assert all(isinstance(p, ColumnProfile) for p in profiles)
    
    # Kiểm tra chứa đủ các cột kể cả ID và IGNORE
    col_names = [p.name for p in profiles]
    assert "id_col" in col_names
    assert "const_col" in col_names
    
    # Kiểm tra suggested_target là TargetAnalysis
    assert isinstance(suggested_target, TargetAnalysis)
    assert suggested_target.recommended_target == "target_col"
