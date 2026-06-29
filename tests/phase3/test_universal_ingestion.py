import pytest
import pandas as pd
import io
from backend.app.core.ingestion.parsers import parse_file, ParseResult

@pytest.fixture
def sample_df():
    return pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})

def test_parse_csv(sample_df):
    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
    res = parse_file(csv_bytes, "data.csv")
    assert isinstance(res, ParseResult)
    assert res.detected_format == "csv"
    assert res.df is not None
    assert res.df.shape == (2, 2)
    assert list(res.df.columns) == ["col1", "col2"]

def test_parse_tsv(sample_df):
    tsv_bytes = sample_df.to_csv(index=False, sep="\t").encode("utf-8")
    res = parse_file(tsv_bytes, "data.tsv")
    assert res.detected_format == "tsv"
    assert res.df.shape == (2, 2)
    assert list(res.df.columns) == ["col1", "col2"]

def test_parse_parquet(sample_df):
    pq_buf = io.BytesIO()
    sample_df.to_parquet(pq_buf, index=False)
    parquet_bytes = pq_buf.getvalue()
    
    res = parse_file(parquet_bytes, "data.parquet")
    assert res.detected_format == "parquet"
    assert res.df.shape == (2, 2)

def test_parse_json(sample_df):
    json_bytes = sample_df.to_json(orient="records").encode("utf-8")
    res = parse_file(json_bytes, "data.json")
    assert res.detected_format == "json"
    assert res.df.shape == (2, 2)

def test_parse_excel_single_sheet(sample_df):
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        sample_df.to_excel(writer, sheet_name="OnlySheet", index=False)
    excel_bytes = excel_buf.getvalue()

    res = parse_file(excel_bytes, "data.xlsx")
    assert res.detected_format == "xlsx"
    assert res.df is not None
    assert res.df.shape == (2, 2)
    assert res.selected_sheet == "OnlySheet"
    assert res.sheets is None

def test_parse_excel_multiple_sheets(sample_df):
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        sample_df.to_excel(writer, sheet_name="SheetA", index=False)
        sample_df.to_excel(writer, sheet_name="SheetB", index=False)
    excel_bytes = excel_buf.getvalue()

    # 1. No sheet_name provided -> returns sheets list and df=None
    res = parse_file(excel_bytes, "data.xlsx")
    assert res.detected_format == "xlsx"
    assert res.df is None
    assert res.sheets == ["SheetA", "SheetB"]
    assert res.selected_sheet is None

    # 2. Provide sheet_name -> reads correct sheet
    res_a = parse_file(excel_bytes, "data.xlsx", sheet_name="SheetA")
    assert res_a.df is not None
    assert res_a.df.shape == (2, 2)
    assert res_a.selected_sheet == "SheetA"

    # 3. Invalid sheet_name -> raises ValueError
    with pytest.raises(ValueError, match="Sheet 'SheetC' not found"):
        parse_file(excel_bytes, "data.xlsx", sheet_name="SheetC")

def test_parse_ods_single_sheet(sample_df):
    ods_buf = io.BytesIO()
    # odfpy is used for writing ODS via pandas
    with pd.ExcelWriter(ods_buf, engine="odf") as writer:
        sample_df.to_excel(writer, sheet_name="ODSSheet", index=False)
    ods_bytes = ods_buf.getvalue()

    res = parse_file(ods_bytes, "data.ods")
    assert res.detected_format == "ods"
    assert res.df is not None
    assert res.df.shape == (2, 2)
    assert res.selected_sheet == "ODSSheet"

def test_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported file format"):
        parse_file(b"content", "data.png")

def test_corrupt_file():
    with pytest.raises(ValueError, match="Failed to parse corrupt"):
        parse_file(b"corrupt-data", "data.parquet")
