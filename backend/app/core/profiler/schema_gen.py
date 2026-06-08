import pandera as pa
from typing import List
from ..profiler.column_profile import ColumnProfile

def generate_pandera_schema(profiles: List[ColumnProfile], target_col: str) -> pa.DataFrameSchema:
    """Automatically generates a Pandera schema from ColumnProfiles."""
    columns = {}
    
    for p in profiles:
        if p.name == target_col:
            columns[p.name] = pa.Column(pa.Int, nullable=False, checks=[pa.Check.isin([0, 1])])
            continue
            
        if p.inferred_role == "numeric":
            columns[p.name] = pa.Column(pa.Float, nullable=True)
        elif p.inferred_role == "categorical":
            columns[p.name] = pa.Column(pa.String, nullable=True)
        elif p.inferred_role == "datetime":
            columns[p.name] = pa.Column(pa.DateTime, nullable=True)
        else:
            columns[p.name] = pa.Column(pa.String, nullable=True)
            
    return pa.DataFrameSchema(columns)
