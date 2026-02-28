# Installing Apache Airflow (recommended)

Airflow has a large dependency graph. To avoid resolver conflicts, install it separately using the official constraints file, not from the project's main requirements.txt.

## Python 3.10 example
Replace `<VERSION>` with the Airflow version you want (e.g., `2.10.3`).

```powershell
# Inside your venv
pip install "apache-airflow==<VERSION>" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-<VERSION>/constraints-3.10.txt"
```

This pins all providers and transitive dependencies to versions verified by the Airflow project (e.g., packaging, colorlog, provider bundles).

## Notes
- If you already installed Airflow without constraints, uninstall first:
  ```powershell
  pip uninstall -y apache-airflow apache-airflow-core apache-airflow-providers-* colorlog packaging
  ```
- Then re-install using the constraint command above.
- For other Python versions, change the constraints file suffix accordingly (e.g., `constraints-3.11.txt`).
- To run the DAG added in this repo, Airflow only needs to have access to the project folder on its PYTHONPATH and the project dependencies.
