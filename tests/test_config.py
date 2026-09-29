"""Tests de configuracion: validacion, YAML y overrides."""

from __future__ import annotations

import textwrap

import pytest

from digitclf.config import ExperimentConfig, config_from_dict, load_config, parse_overrides


def test_defaults_are_valid():
    cfg = ExperimentConfig()
    assert cfg.optim.max_epochs == 10
    assert cfg.model.name == "simple_cnn"
    assert 0 < cfg.data.val_split < 1


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(seed=-1)
    with pytest.raises(ValueError):
        ExperimentConfig().replace(**{"data.val_split": 1.5})
    with pytest.raises(ValueError):
        ExperimentConfig().replace(**{"optim.lr": 0})
    with pytest.raises(ValueError):
        ExperimentConfig().replace(**{"optim.label_smoothing": 1.5})
    with pytest.raises(ValueError):
        ExperimentConfig().replace(**{"data.batch_size": 0})


def test_unknown_override_raises():
    with pytest.raises(KeyError):
        ExperimentConfig().replace(**{"no.existe": 1})


def test_type_coercion_in_overrides():
    cfg = ExperimentConfig().replace(**{"optim.lr": "0.005", "data.augment": "false"})
    assert isinstance(cfg.optim.lr, float) and cfg.optim.lr == 0.005
    assert cfg.data.augment is False


def test_parse_overrides():
    out = parse_overrides(["optim.lr=3e-3", "model.name=resnet_mnist", "data.batch_size=256"])
    assert out == {"optim.lr": "3e-3", "model.name": "resnet_mnist", "data.batch_size": "256"}


def test_parse_overrides_invalid():
    with pytest.raises(ValueError):
        parse_overrides(["optim.lr"])


def test_config_from_dict_partial():
    cfg = config_from_dict({"optim": {"max_epochs": 3}})
    assert cfg.optim.max_epochs == 3
    assert cfg.optim.lr == ExperimentConfig().optim.lr


def test_config_from_dict_unknown_section_param():
    with pytest.raises(KeyError):
        config_from_dict({"optim": {"no_existe": 1}})


def test_load_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent("""
        name: test_run
        optim:
          max_epochs: 2
        model:
          name: mlp
    """))
    cfg = load_config(path)
    assert cfg.name == "test_run"
    assert cfg.optim.max_epochs == 2
    assert cfg.model.name == "mlp"


def test_default_yaml_is_valid():
    cfg = load_config()
    assert cfg.name
    assert cfg.data.num_classes == 10


def test_roundtrip_yaml(tmp_path):
    cfg = ExperimentConfig(name="roundtrip")
    path = cfg.save(tmp_path / "out.yaml")
    reloaded = load_config(path)
    assert reloaded.to_dict() == cfg.to_dict()


def test_replace_does_not_mutate_original():
    original = ExperimentConfig()
    modified = original.replace(**{"optim.lr": 0.5})
    assert original.optim.lr != 0.5
    assert modified.optim.lr == 0.5
