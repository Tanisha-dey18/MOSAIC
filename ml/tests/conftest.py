from pathlib import Path

import pytest

from ml.config import ExperimentConfig, FeatureConfig
from ml.data_ingestion.physionet_wearable import load_dataset
from ml.features.extract import extract_features
from ml.tests.synthetic import write_physionet_tree


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    return write_physionet_tree(tmp_path_factory.mktemp("synthetic_physionet"))


@pytest.fixture(scope="session")
def sessions(synthetic_root):
    sess, errors = load_dataset(synthetic_root, protocols=("STRESS",))
    assert errors == []
    return sess


@pytest.fixture(scope="session")
def features(sessions):
    return extract_features(sessions, FeatureConfig())


@pytest.fixture()
def exp_cfg() -> ExperimentConfig:
    cfg = ExperimentConfig(n_folds=3, bootstrap_iterations=50, autoencoder_epochs=15)
    return cfg
