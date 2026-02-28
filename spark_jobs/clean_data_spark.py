import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower
from pyspark.sql.types import IntegerType, DoubleType

# Validate import pandera
try:
    import pandera as pa
    from pandera.typing import DataFrame, Series
except ImportError:
    pa = None

# --- 1. CONFIGURATION & UTILS ---


def setup_windows_env(base_dir):
    """Configure environment variables for Spark on Windows"""
    # Hadoop
    hadoop_path = os.path.join(base_dir, 'bin', 'hadoop')
    os.environ['HADOOP_HOME'] = hadoop_path

    if not os.path.exists(os.path.join(hadoop_path, 'bin', 'winutils.exe')):
        # Warn but don't exit so launcher.py can handle Linux logic
        print(f"⚠️ Warning: winutils.exe not found at {hadoop_path}")

    # Java (Auto-detect)
    adoptium_dir = os.path.join(base_dir, 'bin', 'Eclipse Adoptium')
    try:
        if os.path.exists(adoptium_dir):
            jdk_name = [f for f in os.listdir(
                adoptium_dir) if f.startswith('jdk-11')][0]
            java_path = os.path.join(adoptium_dir, jdk_name)
            os.environ['JAVA_HOME'] = java_path
    except Exception:
        print("⚠️ Warning: Local JDK 11 not found. Using system Java.")

    # Update Path
    paths = [os.path.join(hadoop_path, 'bin')]
    if 'JAVA_HOME' in os.environ:
        paths.append(os.path.join(os.environ['JAVA_HOME'], 'bin'))

    os.environ['PATH'] = os.pathsep.join(paths + [os.environ['PATH']])


def get_paths():
    """Return S3 (MinIO) paths for Spark to process."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # MinIO paths (S3A)
    # Ensure 'telco_churn.parquet' is uploaded to bucket datalake/raw
    input_path = "s3a://datalake/raw/telco_churn.parquet"
    output_path = "s3a://datalake/processed/features"

    # IMPORTANT: This return prevents NoneType errors in callers
    return base_dir, input_path, output_path

# --- 2. TRANSFORMATION LOGIC ---


def clean_dataframe(df):
    """Core data cleaning logic."""

    # A. Normalize column names
    df = df.select([col(c).alias(c.strip().lower().replace(' ', ''))
                   for c in df.columns])
    cols = df.columns

    # B. Handle churn column
    if 'churnvalue' in cols:
        df = df.withColumn("Churn", col("churnvalue").cast(IntegerType()))
    elif 'churnlabel' in cols:
        df = df.withColumn("Churn", when(
            lower(col("churnlabel")) == "yes", 1).otherwise(0))
    elif 'churn' in cols:
        df = df.withColumn("Churn", when(
            col("churn").isin("Yes", "1"), 1).otherwise(0))

    df = df.fillna(0, subset=["Churn"])

    # C. Cast numeric columns
    numeric_cols = {'totalcharges': 'TotalCharges',
                    'monthlycharges': 'MonthlyCharges'}
    for src, dest in numeric_cols.items():
        if src in cols:
            df = df.withColumn(dest, col(src).cast(
                DoubleType())).fillna(0.0, subset=[dest])

    # D. Handle tenure
    tenure_col = 'tenuremonths' if 'tenuremonths' in cols else (
        'tenure' if 'tenure' in cols else None)
    if tenure_col:
        df = df.withColumn("tenure", col(tenure_col).cast(IntegerType()))

    # E. Rename ID
    if 'customerid' in cols:
        df = df.withColumnRenamed("customerid", "customerID")

    # F. Select columns
    required = ['customerID', 'tenure',
                'MonthlyCharges', 'TotalCharges', 'Churn']
    final_cols = [c for c in required if c in df.columns]

    # Create result before validation
    df_result = df.select(*final_cols)

    # G. Validate data
    if pa:
        try:
            print("🔍 Validating data schema...")
            sample_pd = df_result.limit(5).toPandas()
            schema = pa.DataFrameSchema({
                "MonthlyCharges": pa.Column(float, checks=pa.Check.ge(0), required=False),
                "Churn": pa.Column(int, checks=pa.Check.isin([0, 1]), required=False)
            })
            schema.validate(sample_pd)
            print("✅ Data Validation Passed!")
        except Exception as e:
            print(f"⚠️ Validation Warning: {e}")

    return df_result

# --- 3. MAIN EXECUTION ---


def run():
    base_dir, input_path, output_path = get_paths()
    setup_windows_env(base_dir)

    print("🔌 Configuring Spark for MinIO/S3...")

    # Configure Spark + AWS Jars to connect to MinIO
    spark = SparkSession.builder \
        .appName("CDP_Telco_ETL") \
        .master("local[*]") \
        .config("spark.sql.caseSensitive", "false") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "admin") \
        .config("spark.hadoop.fs.s3a.secret.key", "password") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    try:
        print(f"🚀 Reading from MinIO: {input_path}")
        df = spark.read.parquet(input_path)

        df_clean = clean_dataframe(df)

        print(f"💾 Writing to MinIO: {output_path}")
        df_clean.coalesce(1).write.mode("overwrite").parquet(output_path)
        print("✅ SUCCESS! Data saved to Data Lake (MinIO).")

    except Exception as e:
        print(f"❌ ERROR: {e}")
    finally:
        spark.stop()


if __name__ == "__main__":
    run()
