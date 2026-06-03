from importlib import resources
from pathlib import Path

SAMPLE_DATA_ROOT_DIR = (
    Path(str(resources.files("tests"))) / "sample_data" / "data_sources"
)

SUITES_YML_DIR: Path = (
    Path(str(resources.files("tests.integration.suites"))) / "suites_yml"
)
