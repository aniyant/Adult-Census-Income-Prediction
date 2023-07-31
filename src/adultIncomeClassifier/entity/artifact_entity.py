from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    train_data_filepath: Path
    test_data_filepath: Path
