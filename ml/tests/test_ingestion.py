"""Dataset validation and ingestion (uses SYNTHETIC files in the dataset's format)."""
import numpy as np
import pandas as pd
import pytest

from ml.data_ingestion import generic_csv, physionet_wearable
from ml.data_ingestion.empatica import IngestionError, read_ibi, read_signal, read_tags
from ml.tests.synthetic import generic_csv_frame


def test_read_signal_utc_header(tmp_path):
    f = tmp_path / "EDA.csv"
    f.write_text("2020-01-01 10:00:00\n4.0\n0.1\n0.2\n0.3\n0.4\n")
    s = read_signal(f)
    assert s.fs == 4.0 and s.n_samples == 4
    assert s.start == pd.Timestamp("2020-01-01 10:00:00", tz="UTC")
    assert s.duration_s == 1.0


def test_read_signal_unix_header_and_acc(tmp_path):
    f = tmp_path / "ACC.csv"
    f.write_text("1577872800.0,1577872800.0,1577872800.0\n32.0,32.0,32.0\n1,2,64\n1,2,63\n")
    s = read_signal(f)
    assert s.values.shape == (2, 3) and s.channels == ["x", "y", "z"]
    assert s.start.year == 2020


def test_corrupted_rows_become_nan(tmp_path):
    f = tmp_path / "HR.csv"
    f.write_text("2020-01-01 10:00:00\n1.0\n70\nabc\n72\n")
    s = read_signal(f)
    assert np.isnan(s.values[1, 0])
    assert any("non-numeric" in i for i in s.issues)


@pytest.mark.parametrize("content", ["", "2020-01-01 10:00:00\n", "2020-01-01 10:00:00\n-4\n1\n", "not-a-date\n4\n1\n"])
def test_invalid_headers_raise(tmp_path, content):
    f = tmp_path / "EDA.csv"
    f.write_text(content)
    with pytest.raises(IngestionError):
        read_signal(f)


def test_empty_signal_file_reports_issue(tmp_path):
    f = tmp_path / "TEMP.csv"
    f.write_text("2020-01-01 10:00:00\n4.0\n")
    s = read_signal(f)
    assert s.n_samples == 0 and "file contains no samples" in s.issues


def test_tags_and_ibi(tmp_path):
    (tmp_path / "tags.csv").write_text("")
    assert read_tags(tmp_path / "tags.csv") == []
    assert read_tags(tmp_path / "missing.csv") == []
    (tmp_path / "IBI.csv").write_text("2020-01-01 10:00:00,IBI\n1.0,0.8\n2.0,5.0\n3.0,0.1\n")
    _, ibi = read_ibi(tmp_path / "IBI.csv")
    assert len(ibi) == 1  # implausible intervals removed


def test_physionet_phases_and_calibration(sessions):
    by_id = {s.session_id: s for s in sessions}
    v1, v2 = by_id["STRESS_S01"], by_id["STRESS_f01"]
    assert v1.protocol_version == "V1" and v2.protocol_version == "V2"
    assert [p.name for p in v1.phases if p.label == 1] == ["Stroop", "TMCT", "Real Opinion", "Opposite Opinion", "Subtract"]
    assert [p.name for p in v2.phases if p.label == 1] == ["TMCT", "Real Opinion", "Opposite Opinion", "Subtract"]
    assert v1.calibration_end_s == pytest.approx(300) and v2.calibration_end_s == pytest.approx(300)
    # HR starts 10 s after the other signals; offsets must reflect that.
    assert v1.offset_s("HR") == pytest.approx(10)


def test_documented_exclusions_applied(sessions):
    s02 = next(s for s in sessions if s.session_id == "STRESS_S02")
    assert s02.excluded and "duplicated" in s02.exclusion_reason
    assert s02.phases == []


def test_wrong_tag_count_is_unlabelled(synthetic_root, tmp_path):
    import shutil
    dst = tmp_path / "root"
    shutil.copytree(synthetic_root, dst)
    tags = dst / "Wearable_Dataset" / "STRESS" / "S01" / "tags.csv"
    tags.write_text("\n".join(tags.read_text().splitlines()[:5]) + "\n")
    s = physionet_wearable.load_session(dst / "Wearable_Dataset" / "STRESS" / "S01", "STRESS")
    assert s.phases == [] and s.calibration_end_s is None
    assert any("expected 13 tags" in n for n in s.notes)


def test_missing_modality_file(synthetic_root, tmp_path):
    import shutil
    dst = tmp_path / "root"
    shutil.copytree(synthetic_root, dst)
    (dst / "Wearable_Dataset" / "STRESS" / "f02" / "TEMP.csv").unlink()
    s = physionet_wearable.load_session(dst / "Wearable_Dataset" / "STRESS" / "f02", "STRESS")
    assert "TEMP" not in s.signals and any("TEMP.csv missing" in n for n in s.notes)


def test_resolve_root_rejects_unrelated_dir(tmp_path):
    with pytest.raises(IngestionError):
        physionet_wearable.resolve_root(tmp_path)


def test_stress_levels(synthetic_root):
    lv = physionet_wearable.load_stress_levels(synthetic_root)
    assert set(lv.columns) == {"participant_id", "phase", "level"}
    assert (lv["participant_id"] == "S01").any()


def test_generic_csv_validation_and_load():
    df = generic_csv_frame()
    rep = generic_csv.validate_frame(df)
    assert rep["valid"] and rep["participants"] == 2
    sessions = generic_csv.load_frame(df)
    assert len(sessions) == 2
    s = sessions[0]
    assert s.signals["HR"].fs == pytest.approx(1.0)
    assert s.calibration_end_s is not None and s.phases


def test_generic_csv_errors():
    rep = generic_csv.validate_frame(pd.DataFrame({"time": [1, 2], "eda": [1, 2]}))
    assert not rep["valid"] and any("timestamp" in e for e in rep["errors"])
    rep = generic_csv.validate_frame(pd.DataFrame({"timestamp": ["x", "y"], "participant_id": ["a", "a"], "hr": [1, 2]}))
    assert not rep["valid"]
    rep = generic_csv.validate_frame(pd.DataFrame(columns=["timestamp", "participant_id", "hr"]))
    assert not rep["valid"]


def test_generic_csv_duplicates_and_gaps():
    df = generic_csv_frame(1, minutes=6)
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # duplicate timestamp
    df = df.drop(index=range(100, 120))  # 20 s gap
    rep = generic_csv.validate_frame(df)
    assert any("duplicate" in w for w in rep["warnings"])
    s = generic_csv.load_frame(df)[0]
    hr = s.signals["HR"].values[:, 0]
    assert np.isnan(hr[100:120]).all()  # gap re-gridded as NaN, not compressed
