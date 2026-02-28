import sys
import os
import platform

# Ensure Python can locate project modules
sys.path.append(os.getcwd())

# Import script gốc
try:
    import spark_jobs.clean_data_spark as original_script
except ImportError:
    # Fallback: adjust path for import resolution
    sys.path.append(os.path.dirname(os.getcwd()))
    import spark_jobs.clean_data_spark as original_script

# Dummy replacement for Windows-only setup function


def dummy_setup(base_dir):
    print(f"🐧 Detected {platform.system()}. Bypassing Windows Setup.")
    return


# LOGIC MONKEY PATCHING
if platform.system() != "Windows":
    print(f"⚙️ Applying cross-platform patch for {platform.system()}...")
    # Override Windows setup function with no-op on non-Windows
    original_script.setup_windows_env = dummy_setup
else:
    print("🪟 Windows detected. Using original configuration.")

if __name__ == "__main__":
    original_script.run()
