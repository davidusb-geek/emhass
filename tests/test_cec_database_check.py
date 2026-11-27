"""scripts/save_pvlib_module_inverter_database.py: the --check comparison of the
committed CEC databases with a rebuild, and writing only the tables that changed."""

import bz2
import importlib.util
import json
import pathlib
import pickle

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "save_pvlib_module_inverter_database",
    ROOT / "scripts/save_pvlib_module_inverter_database.py",
)
cecdb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cecdb)


def _table(**models) -> pd.DataFrame:
    """A small table shaped like the CEC databases: parameter rows, one column per model,
    plain object values and labels."""
    models = models or {"Maker_A_100": ["Maker A", 0.5, 10], "Maker_B_200": ["Maker B", 0.25, 20]}
    return pd.DataFrame(
        list(zip(*models.values(), strict=True)),
        index=pd.Index(["Manufacturer", "alpha_sc", "N_s"], dtype=object),
        columns=pd.Index(list(models), dtype=object),
        dtype=object,
    )


def _write(path: pathlib.Path, obj, protocol=None) -> None:
    with bz2.BZ2File(path, "w") as f:
        pickle.dump(obj, f, protocol=protocol)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the script at a temporary data folder and source file."""
    monkeypatch.setattr(cecdb, "data_path", tmp_path)
    monkeypatch.setattr(cecdb, "source_path", tmp_path / "cec_database_source.json")
    return tmp_path


@pytest.mark.parametrize("name", ["cec_modules", "cec_inverters"])
def test_the_committed_databases_can_be_read_for_comparison(name):
    table = cecdb.read_committed_database(ROOT / f"src/emhass/data/{name}.pbz2")
    assert isinstance(table, pd.DataFrame)
    assert len(table.index) > 10
    assert table.shape[1] > 1000


def test_the_same_table_compares_equal_whatever_the_pickle_bytes(data_dir):
    """Content decides, not bytes: another pickle protocol (as a pandas upgrade
    would give) is still a match."""
    _write(data_dir / "a.pbz2", _table(), protocol=4)
    _write(data_dir / "b.pbz2", _table(), protocol=5)
    assert (data_dir / "a.pbz2").read_bytes() != (data_dir / "b.pbz2").read_bytes()
    a = cecdb.read_committed_database(data_dir / "a.pbz2")
    b = cecdb.read_committed_database(data_dir / "b.pbz2")
    assert cecdb.describe_differences("cec_modules", a, b) == []


def test_added_removed_and_changed_models_are_named():
    committed = _table()
    rebuilt = _table(
        Maker_A_100=["Maker A", 0.6, 10],  # changed value
        Maker_C_300=["Maker C", 0.1, 30],  # new model; Maker_B_200 dropped
    )
    report = "\n".join(cecdb.describe_differences("cec_modules", committed, rebuilt))
    assert "1 model(s) only in rebuilt, e.g. ['Maker_C_300']" in report
    assert "1 model(s) only in committed, e.g. ['Maker_B_200']" in report
    assert "1 model(s) with changed values, e.g. ['Maker_A_100']" in report


def test_a_new_parameter_row_is_named():
    committed = _table()
    rebuilt = pd.concat(
        [committed, pd.DataFrame([[1, 2]], index=["gamma_r"], columns=committed.columns)]
    )
    report = "\n".join(cecdb.describe_differences("cec_modules", committed, rebuilt))
    assert "only in rebuilt ['gamma_r'], only in committed []" in report
    assert "changed values" not in report
    assert "other types" not in report


def test_a_new_parameter_row_does_not_hide_a_changed_value():
    committed = _table()
    rebuilt = _table(Maker_A_100=["Maker A", 0.5, 11], Maker_B_200=["Maker B", 0.25, 20])
    rebuilt = pd.concat(
        [rebuilt, pd.DataFrame([[1, 2]], index=["gamma_r"], columns=rebuilt.columns)]
    )
    report = "\n".join(cecdb.describe_differences("cec_modules", committed, rebuilt))
    assert "only in rebuilt ['gamma_r'], only in committed []" in report
    assert "1 model(s) with changed values, e.g. ['Maker_A_100']" in report


def test_a_reordered_table_is_reported():
    committed = _table()
    rebuilt = committed[list(reversed(committed.columns))]
    report = "\n".join(cecdb.describe_differences("cec_modules", committed, rebuilt))
    assert "different column order" in report


def test_a_file_that_is_not_a_plain_table_is_reported(data_dir):
    _write(data_dir / "list.pbz2", [1, 2, 3])
    with pytest.raises(pickle.UnpicklingError, match="does not hold a DataFrame"):
        cecdb.read_committed_database(data_dir / "list.pbz2")
    dated = pd.DataFrame({"Maker_A_100": [pd.Timestamp("2024-01-01")]})
    _write(data_dir / "dated.pbz2", dated)
    with pytest.raises(pickle.UnpicklingError, match="not one of the table types"):
        cecdb.read_committed_database(data_dir / "dated.pbz2")


def test_check_passes_on_a_match_and_fails_on_a_difference(data_dir, monkeypatch):
    tables = {"cec_modules": _table(), "cec_inverters": _table()}
    for name, table in tables.items():
        _write(data_dir / f"{name}.pbz2", table)
    monkeypatch.setattr(cecdb, "build_databases", lambda sam_dir: dict(tables))
    assert cecdb.check_databases(data_dir) == 0

    changed = dict(tables, cec_inverters=_table(Maker_A_100=["Maker A", 0.9, 10]))
    monkeypatch.setattr(cecdb, "build_databases", lambda sam_dir: changed)
    assert cecdb.check_databases(data_dir) == 1


def test_check_fails_when_a_committed_file_is_missing(data_dir, monkeypatch):
    _write(data_dir / "cec_modules.pbz2", _table())
    monkeypatch.setattr(
        cecdb,
        "build_databases",
        lambda sam_dir: {"cec_modules": _table(), "cec_inverters": _table()},
    )
    assert cecdb.check_databases(data_dir) == 1


def test_check_writes_nothing(data_dir, monkeypatch):
    _write(data_dir / "cec_modules.pbz2", _table())
    _write(data_dir / "cec_inverters.pbz2", _table())
    before = {p.name: p.read_bytes() for p in data_dir.iterdir()}
    changed = {"cec_modules": _table(Maker_Z=["Z", 1, 1]), "cec_inverters": _table()}
    monkeypatch.setattr(cecdb, "build_databases", lambda sam_dir: changed)
    assert cecdb.main(["--check", "--sam-dir", str(data_dir), "--sam-commit", "0" * 40]) == 1
    assert {p.name: p.read_bytes() for p in data_dir.iterdir()} == before


def test_writing_keeps_unchanged_tables_and_records_the_source(data_dir, monkeypatch):
    """Only a table whose content changed is rewritten, and only then is the SAM
    commit recorded, so the committed files and the recorded source stay paired."""
    _write(data_dir / "cec_modules.pbz2", _table(), protocol=4)
    _write(data_dir / "cec_inverters.pbz2", _table(), protocol=4)
    modules_bytes = (data_dir / "cec_modules.pbz2").read_bytes()
    inverters_bytes = (data_dir / "cec_inverters.pbz2").read_bytes()
    source = data_dir / "cec_database_source.json"
    old_source = {"sam_commit": "a" * 40, "pandas_version": "0.0", "pvlib_version": "0.0"}
    source.write_text(json.dumps(old_source), encoding="utf-8")

    same = {"cec_modules": _table(), "cec_inverters": _table()}
    monkeypatch.setattr(cecdb, "build_databases", lambda sam_dir: same)
    assert cecdb.main(["--sam-dir", str(data_dir), "--sam-commit", "b" * 40]) == 0
    assert (data_dir / "cec_modules.pbz2").read_bytes() == modules_bytes
    assert (data_dir / "cec_inverters.pbz2").read_bytes() == inverters_bytes
    assert json.loads(source.read_text(encoding="utf-8")) == old_source

    # Same values but 10 -> 10.0 is a change too, and is written.
    retyped = dict(
        same,
        cec_inverters=_table(Maker_A_100=["Maker A", 0.5, 10.0], Maker_B_200=["Maker B", 0.25, 20]),
    )
    monkeypatch.setattr(cecdb, "build_databases", lambda sam_dir: retyped)
    assert cecdb.main(["--sam-dir", str(data_dir), "--sam-commit", "c" * 40]) == 0
    assert (data_dir / "cec_modules.pbz2").read_bytes() == modules_bytes
    assert (data_dir / "cec_inverters.pbz2").read_bytes() != inverters_bytes
    written = cecdb.read_committed_database(data_dir / "cec_inverters.pbz2")
    assert cecdb.describe_differences("cec_inverters", written, retyped["cec_inverters"]) == []
    recorded = json.loads(source.read_text(encoding="utf-8"))
    assert recorded == {
        "sam_commit": "c" * 40,
        "pandas_version": cecdb.pd.__version__,
        "pvlib_version": cecdb.pvlib.__version__,
    }


def test_the_source_is_recorded_when_there_is_none_yet(data_dir, monkeypatch):
    tables = {"cec_modules": _table(), "cec_inverters": _table()}
    for name, table in tables.items():
        _write(data_dir / f"{name}.pbz2", table)
    monkeypatch.setattr(cecdb, "build_databases", lambda sam_dir: tables)
    assert cecdb.main(["--sam-dir", str(data_dir), "--sam-commit", "e" * 40]) == 0
    source = data_dir / "cec_database_source.json"
    assert json.loads(source.read_text(encoding="utf-8"))["sam_commit"] == "e" * 40


def test_the_recorded_source_names_a_full_commit_and_both_versions():
    source = json.loads((ROOT / "scripts/cec_database_source.json").read_text(encoding="utf-8"))
    assert set(source) == {"sam_commit", "pandas_version", "pvlib_version"}
    assert len(source["sam_commit"]) == 40
    assert all(c in "0123456789abcdef" for c in source["sam_commit"])


def test_writing_into_an_empty_folder_writes_both_and_records_the_source(data_dir, monkeypatch):
    tables = {"cec_modules": _table(), "cec_inverters": _table()}
    monkeypatch.setattr(cecdb, "build_databases", lambda sam_dir: tables)
    assert cecdb.main(["--sam-dir", str(data_dir), "--sam-commit", "d" * 40]) == 0
    for name, table in tables.items():
        assert cecdb.read_committed_database(data_dir / f"{name}.pbz2").equals(table)
    source = data_dir / "cec_database_source.json"
    assert json.loads(source.read_text(encoding="utf-8"))["sam_commit"] == "d" * 40
    assert b"\r" not in source.read_bytes()


def test_a_table_the_check_could_not_read_is_never_written(data_dir, monkeypatch):
    dated = _table(Maker_A_100=["Maker A", pd.Timestamp("2024-01-01"), 10])
    monkeypatch.setattr(
        cecdb, "build_databases", lambda sam_dir: {"cec_modules": dated, "cec_inverters": _table()}
    )
    with pytest.raises(pickle.UnpicklingError, match="not one of the table types"):
        cecdb.main(["--sam-dir", str(data_dir), "--sam-commit", "f" * 40])
    assert list(data_dir.iterdir()) == []


def test_the_same_values_with_other_types_do_not_match():
    """DataFrame.equals takes 60 and 60.0 as equal; the check does not."""
    committed = _table()
    rebuilt = _table(Maker_A_100=["Maker A", 0.5, 10.0], Maker_B_200=["Maker B", 0.25, 20])
    assert committed.equals(rebuilt)
    report = "\n".join(cecdb.describe_differences("cec_modules", committed, rebuilt))
    assert "1 model(s) with the same values but other types" in report
    assert "Maker_A_100" in report


@pytest.mark.parametrize("axis", ["index", "columns"])
def test_labels_of_another_dtype_do_not_match(axis):
    rebuilt = _table()
    setattr(rebuilt, axis, getattr(rebuilt, axis).astype("string"))
    report = cecdb.describe_differences("cec_modules", _table(), rebuilt)
    assert len(report) == 2, report
    assert "labels have another dtype" in report[1]
    assert cecdb.describe_differences("cec_modules", _table(), cecdb.plain_labels(rebuilt)) == []


def test_a_label_dtype_change_does_not_hide_a_type_only_change():
    rebuilt = _table(Maker_A_100=["Maker A", 0.5, 10.0], Maker_B_200=["Maker B", 0.25, 20])
    rebuilt.index = rebuilt.index.astype("string")
    report = cecdb.describe_differences("cec_modules", _table(), rebuilt)
    assert len(report) == 3, report
    assert "labels have another dtype" in report[1]
    assert "1 model(s) with the same values but other types" in report[2]


def test_a_changed_value_does_not_hide_a_type_only_change():
    committed = _table()
    rebuilt = _table(Maker_A_100=["Maker A", 0.5, 10.0], Maker_B_200=["Maker B", 0.3, 20])
    report = "\n".join(cecdb.describe_differences("cec_modules", committed, rebuilt))
    assert "1 model(s) with changed values, e.g. ['Maker_B_200']" in report
    assert "1 model(s) with the same values but other types" in report
    assert "e.g. ['Maker_A_100']" in report


def test_a_type_only_change_is_found_on_the_rows_both_tables_have():
    committed = _table()
    rebuilt = _table(Maker_A_100=["Maker A", 0.5, 10.0], Maker_B_200=["Maker B", 0.25, 20])
    rebuilt = rebuilt.drop(index="alpha_sc")
    report = "\n".join(cecdb.describe_differences("cec_modules", committed, rebuilt))
    assert "only in rebuilt [], only in committed ['alpha_sc']" in report
    assert "changed values" not in report
    assert "1 model(s) with the same values but other types" in report


def test_the_build_keeps_labels_as_plain_objects(tmp_path, monkeypatch):
    """pandas 3 reads the labels as its string type; the built tables use plain
    objects like the committed files, so they can be read with pandas 2.x too."""

    def fake_retrieve_sam(name=None, path=None):
        table = _table(**{f"{name or pathlib.Path(path).stem}": ["M", 1.0, 2]})
        table.index = table.index.astype("string")
        table.columns = table.columns.astype("string")
        return table

    monkeypatch.setattr(cecdb.pvlib.pvsystem, "retrieve_sam", fake_retrieve_sam)
    for table in cecdb.build_databases(tmp_path).values():
        assert table.index.dtype == object
        assert table.columns.dtype == object
        assert table.shape == (3, 3)


@pytest.mark.parametrize("sha", ["984b44c", "a" * 41, "A" * 40, ""])
def test_a_sam_commit_that_is_not_a_full_sha_is_refused(data_dir, sha):
    with pytest.raises(SystemExit):
        cecdb.main(["--sam-dir", str(data_dir), "--sam-commit", sha])


def test_a_file_that_is_not_a_pickle_is_reported(data_dir, monkeypatch):
    for name in cecdb.database_names:
        (data_dir / f"{name}.pbz2").write_bytes(bz2.compress(b"not a pickle"))
    monkeypatch.setattr(
        cecdb,
        "build_databases",
        lambda sam_dir: {"cec_modules": _table(), "cec_inverters": _table()},
    )
    assert cecdb.check_databases(data_dir) == 1


@pytest.mark.parametrize("recorded", ["0.0", None])
def test_a_pandas_version_mismatch_is_pointed_out(data_dir, monkeypatch, caplog, recorded):
    source = {"sam_commit": "a" * 40, "pandas_version": recorded or cecdb.pd.__version__}
    (data_dir / "cec_database_source.json").write_text(json.dumps(source), encoding="utf-8")
    for name in cecdb.database_names:
        _write(data_dir / f"{name}.pbz2", _table())
    changed = {"cec_modules": _table(Maker_Z=["Z", 1, 1]), "cec_inverters": _table()}
    monkeypatch.setattr(cecdb, "build_databases", lambda sam_dir: changed)
    with caplog.at_level("ERROR"):
        assert cecdb.check_databases(data_dir) == 1
    assert ("were recorded with pandas" in caplog.text) == (recorded is not None)
