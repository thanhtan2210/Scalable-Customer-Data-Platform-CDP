import pandera as pa
import json
from typing import List, Tuple
from datetime import datetime

from ..profiler.column_profile import ColumnProfile, DataRole
from ..storage import storage

def generate_schema(profiles: List[ColumnProfile], dataset_id: str, target_col: str) -> Tuple[pa.DataFrameSchema, dict]:
    """
    Generates a Pandera schema and metadata dictionary based on column profiles.
    """
    columns = {}
    metadata_cols = {}
    
    for p in profiles:
        if p.inferred_role in [DataRole.ID, DataRole.IGNORE, DataRole.TARGET] or p.potential_leakage:
            continue
            
        # Determine pandera data type (NUMERIC is always pa.Float to handle NaNs)
        pa_dtype = pa.String
        if p.inferred_role == DataRole.NUMERIC:
            pa_dtype = pa.Float
        elif p.inferred_role == DataRole.DATETIME:
            pa_dtype = pa.DateTime
            
        columns[p.name] = pa.Column(pa_dtype, nullable=True, coerce=True)
        
        # Populate metadata
        metadata_cols[p.name] = {
            "inferred_role": p.inferred_role.value if hasattr(p.inferred_role, "value") else str(p.inferred_role),
            "transform_strategy": p.transform_strategy,
            "impute_strategy": p.impute_strategy,
            "potential_leakage": p.potential_leakage
        }
        
    schema = pa.DataFrameSchema(columns=columns)
    
    metadata = {
        "dataset_id": dataset_id,
        "target_col": target_col,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "columns": metadata_cols
    }
    
    return schema, metadata

def save_schema(schema: pa.DataFrameSchema, metadata: dict, dataset_id: str, target_col: str) -> Tuple[str, str]:
    """
    Saves the schema and metadata to storage.
    """
    schema_json = schema.to_json()
    metadata_json = json.dumps(metadata, indent=2)
    
    base_path = f"ml_artifacts/{dataset_id}/{target_col}"
    schema_path = f"{base_path}/schema.json"
    metadata_path = f"{base_path}/metadata.json"
    
    storage.upload_file(schema_json.encode('utf-8'), schema_path)
    storage.upload_file(metadata_json.encode('utf-8'), metadata_path)
    
    return schema_path, metadata_path

def load_schema(dataset_id: str, target_col: str) -> Tuple[pa.DataFrameSchema, dict]:
    """
    Loads the schema and metadata from storage.
    """
    base_path = f"ml_artifacts/{dataset_id}/{target_col}"
    schema_path = f"{base_path}/schema.json"
    metadata_path = f"{base_path}/metadata.json"
    
    schema_bytes = storage.download_file(schema_path)
    metadata_bytes = storage.download_file(metadata_path)
    
    schema = pa.DataFrameSchema.from_json(schema_bytes.decode('utf-8'))
    metadata = json.loads(metadata_bytes.decode('utf-8'))
    
    return schema, metadata
