"""
setup.py

Run once to scaffold the entire weather_pipeline project structure.

Usage:
    python setup.py
"""

from pathlib import Path

# ── Directory tree ────────────────────────────────────────────────────────────

DIRECTORIES = [
    "config",
    "src/ingestion",
    "src/processing",
    "src/patches",
    "src/storage",
    "pipelines",
    "scripts",
]

# ── __init__.py locations ─────────────────────────────────────────────────────

INIT_FILES = [
    "config/__init__.py",
    "src/__init__.py",
    "src/ingestion/__init__.py",
    "src/processing/__init__.py",
    "src/patches/__init__.py",
    "src/storage/__init__.py",
    "pipelines/__init__.py",
]

# ── Stub file contents ────────────────────────────────────────────────────────

STUB_FILES = {
    "config/pipeline_config.yaml": """\
# ============================================================
# Weather Pipeline Configuration
# ============================================================

pipeline:
  grib_file: "data.grib"
  years: [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

data:
  raw_chunks_per_year: 5
  series_id_batch_size: 1_000_000

temperature:
  dtype: "float16"

coordinates:
  dtype: "float32"
  round_decimals: 2

patches:
  patch_size: 32
  pad_value: 0.0

storage:
  # Set your Google Drive folder IDs here before running
  drive_folder_id:
    raw_weather: "YOUR_RAW_WEATHER_FOLDER_ID"
    patches: "YOUR_PATCHES_FOLDER_ID"
  local_staging_dir: "/tmp/weather_pipeline_staging"
""",

    "requirements.txt": """\
cfgrib
xarray
pandas
numpy
pyarrow
fastparquet
pyyaml
google-api-python-client
google-auth-httplib2
google-auth-oauthlib
""",
}

# ── Scaffold logic ────────────────────────────────────────────────────────────

def create_directories(base: Path) -> None:
    for directory in DIRECTORIES:
        path = base / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"  [dir]  {path}")


def create_init_files(base: Path) -> None:
    for rel_path in INIT_FILES:
        path = base / rel_path
        if not path.exists():
            path.touch()
            print(f"  [init] {path}")
        else:
            print(f"  [skip] {path} already exists")


def create_stub_files(base: Path) -> None:
    for rel_path, content in STUB_FILES.items():
        path = base / rel_path
        if not path.exists():
            path.write_text(content)
            print(f"  [file] {path}")
        else:
            print(f"  [skip] {path} already exists")


def main() -> None:
    base = Path(__file__).parent.resolve()

    print(f"\nScaffolding weather_pipeline under: {base}\n")

    print("Creating directories...")
    create_directories(base)

    print("\nCreating __init__.py files...")
    create_init_files(base)

    print("\nCreating stub config and requirements files...")
    create_stub_files(base)

    print("\n✓ Project structure ready.")
    print("\nNext steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. Fill in Drive folder IDs in config/pipeline_config.yaml")
    print("  3. export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json")
    print("  4. python -m pipelines.data_pipeline\n")


if __name__ == "__main__":
    main()