from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Depends,
    BackgroundTasks,
    Request,
)
from typing import List, Optional, Dict, Any
import pandas as pd
import io
import uuid
import os
import re
from sqlalchemy.orm import Session
from ...db.models import Dataset, Profile, TrainingJob
from ...db.session import get_db  # Assuming session manager exists
from ...api.schemas import (
    DatasetResponse,
    ProfilingResponse,
    TrainingRequest,
    JobResponse,
    ConfirmCompositeRequest,
    ConfirmCompositeResponse,
    SelectSheetRequest,
    ReEvaluateLeakageRequest,
    ReEvaluateLeakageResponse,
)
from ...core.storage import storage
from ...core.profiler.orchestrator import run_profiling
from ...core.profiler.target_analysis import TargetAnalysis
from ...core.training.automl import run_automl
from ...core.ingestion.parsers import parse_file
from ...core.profiler.column_profile import ColumnProfile
from ...core.limiter import limiter

router = APIRouter(prefix="/datasets", tags=["datasets"])


def sanitize_filename(filename: str) -> str:
    # Bỏ path separators
    filename = os.path.basename(filename)
    # Chỉ giữ alphanumeric, dash, underscore, dot
    filename = re.sub(r"[^\w\-_\.]", "_", filename)
    # Giới hạn độ dài
    name, ext = os.path.splitext(filename)
    name = name[:100]
    return f"{name}{ext}"


@router.post("/upload", response_model=DatasetResponse)
@limiter.limit("10/minute")
async def upload_dataset(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    MAX_FILE_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")) * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB",
        )

    safe_name = sanitize_filename(file.filename)

    try:
        result = parse_file(content=content, filename=safe_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    dataset_id = str(uuid.uuid4())
    user_id = "default_user"  # Placeholder for auth

    requires_sheet_selection = (
        result.df is None and result.sheets is not None and len(result.sheets) > 1
    )

    if requires_sheet_selection:
        # Store raw Excel file to R2
        r2_path = f"raw/{user_id}/{dataset_id}/{safe_name}"
        storage.upload_file(content, r2_path)

        new_dataset = Dataset(
            id=dataset_id,
            user_id=user_id,
            filename=safe_name,
            r2_path=r2_path,
            row_count=None,
            col_count=None,
            status="uploaded",
        )
        db.add(new_dataset)
        db.commit()

        return {
            "dataset_id": dataset_id,
            "row_count": None,
            "col_count": None,
            "status": "uploaded",
            "detected_format": result.detected_format,
            "sheets": result.sheets,
            "requires_sheet_selection": True,
            "r2_path": r2_path,
        }

    # If df has data:
    df = result.df
    if len(df.columns) < 2:
        raise HTTPException(
            status_code=400, detail="Dataset must have at least 2 columns"
        )

    # Save to R2 - keep CSV/Parquet as raw, convert other formats to Parquet for downstream compatibility
    if result.detected_format == "csv":
        upload_content = content
        filename_to_save = safe_name
    elif result.detected_format == "parquet":
        upload_content = content
        filename_to_save = safe_name
    else:
        pq_buffer = io.BytesIO()
        df.to_parquet(pq_buffer, index=False)
        upload_content = pq_buffer.getvalue()
        filename_to_save = f"{safe_name.split('.')[0]}.parquet"

    r2_path = f"raw/{user_id}/{dataset_id}/{filename_to_save}"
    storage.upload_file(upload_content, r2_path)

    new_dataset = Dataset(
        id=dataset_id,
        user_id=user_id,
        filename=filename_to_save,
        r2_path=r2_path,
        row_count=len(df),
        col_count=len(df.columns),
        status="uploaded",
    )
    db.add(new_dataset)
    db.commit()

    return {
        "dataset_id": dataset_id,
        "row_count": len(df),
        "col_count": len(df.columns),
        "status": "uploaded",
        "detected_format": result.detected_format,
        "sheets": None,
        "requires_sheet_selection": False,
        "r2_path": r2_path,
    }


@router.post("/{dataset_id}/profile", response_model=ProfilingResponse)
@limiter.limit("20/minute")
async def profile_dataset(
    request: Request, dataset_id: str, db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # 1. Read from R2
    content = storage.download_file(dataset.r2_path)
    result = parse_file(content=content, filename=dataset.filename)
    df = result.df

    # 2. Run Profiling
    profiles, target_analysis = run_profiling(df)

    # 4. Save to DB
    new_profile = Profile(
        dataset_id=dataset_id,
        profiles_json=[p.dict() for p in profiles],
        suggested_target=(
            target_analysis.json()
            if hasattr(target_analysis, "json")
            else str(target_analysis)
        ),
    )
    db.merge(new_profile)
    dataset.status = "profiled"
    db.commit()

    warnings = []
    for p in profiles:
        role_str = (
            p.inferred_role.value
            if hasattr(p.inferred_role, "value")
            else str(p.inferred_role)
        )
        if role_str in ["ID", "IGNORE"]:
            reason = "Identifier column" if role_str == "ID" else "Ignored column"
            warnings.append(f"Column '{p.name}' dropped: {reason}")

    return ProfilingResponse(
        dataset_id=dataset_id,
        profiles=profiles,
        suggested_target=target_analysis.recommended_target,
        candidate_targets=target_analysis.candidate_targets,
        composite_target=target_analysis.composite_target,
        churn_column_group=target_analysis.churn_column_group,
        leakage_suspects=target_analysis.leakage_suspects,
        warnings=warnings,
    )


@router.get("/{dataset_id}/profile", response_model=ProfilingResponse)
async def get_dataset_profile(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    profile = db.query(Profile).filter(Profile.dataset_id == dataset_id).first()
    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Profile not found. Please run profiling first using POST.",
        )

    try:
        from ...core.profiler.target_analysis import TargetAnalysis

        target_analysis = TargetAnalysis.parse_raw(profile.suggested_target)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse target analysis from database: {str(e)}",
        )

    from ...core.profiler.column_profile import ColumnProfile

    profiles = [ColumnProfile(**p) for p in profile.profiles_json]

    warnings = []
    for p in profiles:
        role_str = (
            p.inferred_role.value
            if hasattr(p.inferred_role, "value")
            else str(p.inferred_role)
        )
        if role_str in ["ID", "IGNORE"]:
            reason = "Identifier column" if role_str == "ID" else "Ignored column"
            warnings.append(f"Column '{p.name}' dropped: {reason}")

    return ProfilingResponse(
        dataset_id=dataset_id,
        profiles=profiles,
        suggested_target=target_analysis.recommended_target,
        candidate_targets=target_analysis.candidate_targets,
        composite_target=target_analysis.composite_target,
        churn_column_group=target_analysis.churn_column_group,
        leakage_suspects=target_analysis.leakage_suspects,
        warnings=warnings,
    )


@router.post("/{dataset_id}/confirm-composite", response_model=ConfirmCompositeResponse)
async def confirm_composite(
    dataset_id: str, req: ConfirmCompositeRequest, db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    profile = db.query(Profile).filter(Profile.dataset_id == dataset_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # If confirmed is False, clear composite target in DB and return cpi_attached=False
    if not req.confirmed:
        try:
            target_analysis = TargetAnalysis.parse_raw(profile.suggested_target)
            target_analysis.composite_target = None
            profile.suggested_target = target_analysis.json()
        except Exception:
            profile.suggested_target = None

        db.commit()
        return {
            "dataset_id": dataset_id,
            "composite_target": None,
            "cpi_attached": False,
        }

    # If confirmed is True:
    # 1. Download file from R2
    content = storage.download_file(dataset.r2_path)
    df = (
        pd.read_csv(io.BytesIO(content))
        if dataset.r2_path.endswith(".csv") or dataset.filename.endswith(".csv")
        else pd.read_parquet(io.BytesIO(content))
    )

    # 2. Get target analysis
    try:
        target_analysis = TargetAnalysis.parse_raw(profile.suggested_target)
    except Exception:
        _, target_analysis = run_profiling(df)

    # 3. Determine strategy & source columns
    strategy = req.selected_strategy
    if not strategy:
        if target_analysis.composite_target:
            strategy = target_analysis.composite_target.strategy
        else:
            strategy = "WEIGHTED"

    source_columns = req.selected_source_columns
    if not source_columns:
        if target_analysis.composite_target:
            source_columns = target_analysis.composite_target.source_columns
        else:
            source_columns = [
                c.name
                for c in target_analysis.churn_column_group
                if c.group_role in ("AUXILIARY", "DUPLICATE")
                and c.name != target_analysis.recommended_target
            ]

    if strategy == "NONE" or not source_columns:
        target_analysis.composite_target = None
        profile.suggested_target = target_analysis.json()
        db.commit()
        return {
            "dataset_id": dataset_id,
            "composite_target": None,
            "cpi_attached": False,
        }

    # 4. Perform target synthesis
    from ...core.profiler.target_synthesizer import (
        _pca_synthesis,
        _weighted_synthesis,
        _auto_assign_weights,
    )
    from ...core.profiler.target_analysis import CompositeTargetConfig

    cpi_variance_explained = None
    weights = None
    cpi_series = None

    if strategy == "PCA":
        numeric_cols = [
            c
            for c in source_columns
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(numeric_cols) >= 2:
            cpi_variance_explained, cpi_series = _pca_synthesis(
                df, numeric_cols, target_analysis.recommended_target
            )
        else:
            strategy = "WEIGHTED"  # fallback

    if strategy == "WEIGHTED":
        eligible_items = [
            c for c in target_analysis.churn_column_group if c.name in source_columns
        ]
        weights = _auto_assign_weights(eligible_items, df)
        cpi_series = _weighted_synthesis(df, weights)

    # 5. Build CompositeTargetConfig
    composite_config = CompositeTargetConfig(
        strategy=strategy,
        source_columns=source_columns,
        cpi_variance_explained=cpi_variance_explained,
        weights=weights,
        cpi_column_name="cpi_score",
        requires_confirmation=False,
    )

    # 6. Attach cpi_score and re-upload to processed/... path as parquet
    df["cpi_score"] = cpi_series

    pq_buffer = io.BytesIO()
    df.to_parquet(pq_buffer, index=False)
    pq_content = pq_buffer.getvalue()

    new_r2_path = f"processed/{dataset_id}/with_cpi.parquet"
    storage.upload_file(pq_content, new_r2_path)

    # 7. Update Dataset & Profile records
    dataset.r2_path = new_r2_path
    dataset.filename = "with_cpi.parquet"  # ensures jobs.py reads it as parquet

    target_analysis.composite_target = composite_config
    profile.suggested_target = target_analysis.json()

    # Append cpi_score to ColumnProfile list
    from ...core.profiler.column_profile import ColumnProfile, DataRole

    cpi_profile = ColumnProfile(
        name="cpi_score",
        inferred_dtype="float64",
        inferred_role=DataRole.NUMERIC,
        confidence_score=1.0,
        null_pct=0.0,
        unique_count=df["cpi_score"].nunique(),
        entropy=0.0,
        potential_leakage=False,
        transform_strategy="standard",
        impute_strategy="median",
    )

    existing_profiles = profile.profiles_json or []
    existing_profiles = [p for p in existing_profiles if p.get("name") != "cpi_score"]
    existing_profiles.append(cpi_profile.dict())
    profile.profiles_json = existing_profiles

    db.commit()

    return {
        "dataset_id": dataset_id,
        "composite_target": composite_config,
        "cpi_attached": True,
    }


@router.post("/{dataset_id}/select-sheet", response_model=DatasetResponse)
async def select_sheet(
    dataset_id: str, req: SelectSheetRequest, db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Download raw file from R2
    try:
        content = storage.download_file(dataset.r2_path)
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"File not found in storage: {str(e)}"
        )

    # Parse file with selected sheet_name
    try:
        result = parse_file(
            content=content, filename=dataset.filename, sheet_name=req.sheet_name
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    df = result.df
    if df is None:
        raise HTTPException(
            status_code=400, detail="Sheet not found or could not be parsed"
        )

    if len(df.columns) < 2:
        raise HTTPException(
            status_code=400, detail="Dataset must have at least 2 columns"
        )

    # Convert data to Parquet
    pq_buffer = io.BytesIO()
    df.to_parquet(pq_buffer, index=False)
    pq_content = pq_buffer.getvalue()

    # Save the selected parquet sheet to R2
    user_id = dataset.user_id
    parquet_filename = f"{dataset.filename.split('.')[0]}.parquet"
    new_r2_path = f"raw/{user_id}/{dataset_id}/{parquet_filename}"

    storage.upload_file(pq_content, new_r2_path)

    # Update DB
    dataset.r2_path = new_r2_path
    dataset.filename = parquet_filename
    dataset.status = "uploaded"
    dataset.row_count = len(df)
    dataset.col_count = len(df.columns)
    db.commit()

    return {
        "dataset_id": dataset_id,
        "row_count": len(df),
        "col_count": len(df.columns),
        "status": "uploaded",
        "detected_format": result.detected_format,
        "sheets": None,
        "requires_sheet_selection": False,
    }


@router.post(
    "/{dataset_id}/re-evaluate-leakage", response_model=ReEvaluateLeakageResponse
)
async def re_evaluate_leakage(
    dataset_id: str, req: ReEvaluateLeakageRequest, db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    profile = db.query(Profile).filter(Profile.dataset_id == dataset_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # 1. Load Data from R2
    content = storage.download_file(dataset.r2_path)
    result = parse_file(content=content, filename=dataset.filename)
    df = result.df

    if df is None:
        raise HTTPException(status_code=400, detail="Failed to parse dataset dataframe")

    if req.confirmed_target not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Confirmed target '{req.confirmed_target}' not found in dataset columns",
        )

    # 2. Reset and update roles
    from ...core.profiler.column_profile import DataRole, ColumnProfile
    from ...core.profiler.orchestrator import check_leakage, ROLE_RECIPES

    profiles_dict = profile.profiles_json or []

    for p in profiles_dict:
        col_name = p["name"]
        if col_name == req.confirmed_target:
            p["inferred_role"] = (
                DataRole.TARGET.value
                if hasattr(DataRole.TARGET, "value")
                else str(DataRole.TARGET)
            )
            p["confidence_score"] = 1.0
            p["potential_leakage"] = False
            p["leakage_score"] = None
        else:
            role_val = (
                p["inferred_role"].value
                if hasattr(p["inferred_role"], "value")
                else str(p["inferred_role"])
            )
            if role_val == "TARGET" or role_val == DataRole.TARGET:
                # Revert old target to a sensible default role (Numeric or Categorical) based on data type
                is_numeric = p["inferred_dtype"] in (
                    "float64",
                    "int64",
                    "float32",
                    "int32",
                )
                p["inferred_role"] = (
                    DataRole.NUMERIC.value if is_numeric else DataRole.CATEGORICAL.value
                )
                p["confidence_score"] = 0.8
            p["potential_leakage"] = False
            p["leakage_score"] = None

    # 3. Re-evaluate leakage
    check_leakage(req.confirmed_target, profiles_dict, df)

    # 4. Update recipes & reconstruct ColumnProfile objects
    updated_profiles = []
    for p in profiles_dict:
        role = p["inferred_role"]
        try:
            role_enum = DataRole(role)
        except ValueError:
            role_enum = DataRole.IGNORE
        recipe = ROLE_RECIPES.get(role_enum, ROLE_RECIPES[DataRole.IGNORE])
        if recipe:
            p["impute_strategy"] = recipe["impute"]
            p["transform_strategy"] = recipe["transform"]

        # Pydantic safety
        p.setdefault("regex_pattern", None)
        p.setdefault("mean_length", None)
        p.setdefault("leakage_score", None)
        p.setdefault("potential_leakage", False)

        updated_profiles.append(ColumnProfile(**p))

    # 5. Save back to DB
    profile.profiles_json = [up.dict() for up in updated_profiles]

    try:
        target_analysis = TargetAnalysis.parse_raw(profile.suggested_target)
        target_analysis.recommended_target = req.confirmed_target
        profile.suggested_target = target_analysis.json()
    except Exception:
        pass

    db.commit()

    return ReEvaluateLeakageResponse(
        profiles_updated_in_db=True,
        updated_profiles=updated_profiles,
        dataset_id=dataset_id,
        confirmed_target=req.confirmed_target,
        leakage_suspects=[p.name for p in updated_profiles if p.potential_leakage],
    )
