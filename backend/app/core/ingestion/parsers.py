import io
import pandas as pd
from pydantic import BaseModel
from typing import Any, List, Optional


class ParseResult(BaseModel):
    df: Any
    detected_format: str
    sheets: Optional[List[str]] = None
    selected_sheet: Optional[str] = None
    warnings: List[str] = []

    class Config:
        arbitrary_types_allowed = True


def parse_file(
    content: bytes, filename: str, sheet_name: Optional[str] = None
) -> ParseResult:
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    supported_formats = {
        "csv": "csv",
        "tsv": "tsv",
        "parquet": "parquet",
        "json": "json",
        "xlsx": "xlsx",
        "xls": "xlsx",
        "ods": "ods",
    }

    if ext not in supported_formats:
        raise ValueError(
            f"Unsupported file format: .{ext}"
            if ext
            else "File has no extension and format could not be determined"
        )

    detected_format = supported_formats[ext]

    try:
        if detected_format == "csv":
            df = pd.read_csv(io.BytesIO(content))
            return ParseResult(df=df, detected_format="csv")

        elif detected_format == "tsv":
            df = pd.read_csv(io.BytesIO(content), sep="\t")
            return ParseResult(df=df, detected_format="tsv")

        elif detected_format == "parquet":
            df = pd.read_parquet(io.BytesIO(content))
            return ParseResult(df=df, detected_format="parquet")

        elif detected_format == "json":
            df = pd.read_json(io.BytesIO(content))
            return ParseResult(df=df, detected_format="json")

        elif detected_format in ("xlsx", "ods"):
            engine = "odf" if detected_format == "ods" else None

            # Read sheets
            excel_file = pd.ExcelFile(io.BytesIO(content), engine=engine)
            sheet_names = excel_file.sheet_names

            # 1. If sheet_name is provided
            if sheet_name is not None:
                if sheet_name not in sheet_names:
                    raise ValueError(
                        f"Sheet '{sheet_name}' not found in file. Available sheets: {sheet_names}"
                    )
                df = pd.read_excel(
                    io.BytesIO(content), sheet_name=sheet_name, engine=engine
                )
                return ParseResult(
                    df=df, detected_format=detected_format, selected_sheet=sheet_name
                )

            # 2. If sheet_name is not provided
            if len(sheet_names) == 1:
                selected_sheet = sheet_names[0]
                df = pd.read_excel(
                    io.BytesIO(content), sheet_name=selected_sheet, engine=engine
                )
                return ParseResult(
                    df=df,
                    detected_format=detected_format,
                    selected_sheet=selected_sheet,
                )
            else:
                # Multiple sheets, don't read data, ask client to select
                return ParseResult(
                    df=None, detected_format=detected_format, sheets=sheet_names
                )

    except Exception as e:
        if isinstance(e, ValueError) and (
            "Unsupported file format" in str(e) or "not found in file" in str(e)
        ):
            raise e
        raise ValueError(
            f"Failed to parse corrupt or invalid {detected_format.upper()} file: {str(e)}"
        )
