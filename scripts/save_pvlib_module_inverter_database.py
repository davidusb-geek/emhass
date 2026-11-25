"""
This is a script for saving the database to be used by PVLib for
modules and inverters models. This was necessary to keep the
database up to date with the latest database version from SAM
while updating the out-dated original database from PVLib.
This script uses the tabulate package: pip install tabulate

Usage:
    # Build from "CEC Modules.csv" and "CEC Inverters.csv" copied into src/emhass/data
    python scripts/save_pvlib_module_inverter_database.py
    # Build from a SAM checkout and record the SAM commit in cec_database_source.json
    python scripts/save_pvlib_module_inverter_database.py \\
        --sam-dir /tmp/SAM/deploy/libraries --sam-commit <sha>
    # Rebuild in memory and compare with the committed databases (no files written)
    python scripts/save_pvlib_module_inverter_database.py --check --sam-dir <dir>

The SAM commit (https://github.com/NatLabRockies/SAM, deploy/libraries) and the pandas
and pvlib versions a rebuild needs to reproduce the committed databases are recorded in
scripts/cec_database_source.json (pandas matters too: pandas 2 and 3 read some long
floats in the CSVs one digit apart). The check rebuilds from those and compares the decoded
tables, since the pickled bytes change with the pandas version.
"""

import argparse
import bz2
import io
import json
import logging
import pathlib
import pickle as cPickle
import re
import sys

import pandas as pd
import pvlib

# the root folder
root = pathlib.Path(__file__).resolve().parents[1]
data_path = root / "src/emhass/data/"  # NOTE CUSTOM DATA PATH
source_path = root / "scripts/cec_database_source.json"
database_names = ("cec_modules", "cec_inverters")

# create logger
logger = logging.getLogger(__name__)

strait_str = "================="

# The committed databases are plain pandas DataFrames of strings and numbers with plain
# (object) row and column labels. Reading them for the comparison only needs the classes
# such a table pickles to; anything else means the file is not the table this script
# writes, and the check reports that instead of comparing. pandas 2.x and 3.x spell the
# DataFrame and Index classes differently, so both spellings are listed.
TABLE_CLASSES = {
    ("builtins", "slice"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("pandas", "DataFrame"),
    ("pandas", "Index"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.internals.managers", "BlockManager"),
}

# The usual errors from reading a missing, truncated or unexpected file.
READ_ERRORS = (
    OSError,
    EOFError,
    ValueError,
    TypeError,
    AttributeError,
    ImportError,
    cPickle.UnpicklingError,
)


class TableUnpickler(cPickle.Unpickler):
    """Read a committed database, accepting only the classes in TABLE_CLASSES."""

    def find_class(self, module, name):
        if (module, name) not in TABLE_CLASSES:
            raise cPickle.UnpicklingError(
                f"{module}.{name} is not one of the table types this script writes"
            )
        return super().find_class(module, name)


def read_table(data: bytes, label: str) -> pd.DataFrame:
    """Decode a pickled database table for comparison."""
    table = TableUnpickler(io.BytesIO(data)).load()
    if not isinstance(table, pd.DataFrame):
        raise cPickle.UnpicklingError(f"{label} does not hold a DataFrame")
    return table


def read_committed_database(path: pathlib.Path) -> pd.DataFrame:
    """Decode a committed .pbz2 database for comparison."""
    with bz2.BZ2File(path, "rb") as f:
        return read_table(f.read(), path.name)


def plain_labels(table: pd.DataFrame) -> pd.DataFrame:
    """Keep the row and column labels as plain objects, as in the committed files.

    pandas 3 reads the labels as its own string type, and a table pickled with that
    type cannot be read with pandas 2.x, which pyproject still allows.
    """
    table = table.copy()
    table.index = table.index.astype(object)
    table.columns = table.columns.astype(object)
    return table


def append_missing_columns(base: pd.DataFrame, extra: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Append the columns (models) of extra that base does not have yet."""
    cols_to_keep = [elem for elem in extra.columns if elem not in base.columns]
    return pd.concat([base, extra[cols_to_keep]], axis=1), cols_to_keep


def build_databases(sam_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    """Build the modules and inverters databases from SAM, pvlib and the EMHASS CSVs."""
    logger.info("Reading original outdated database from PVLib")
    cec_modules_0 = pvlib.pvsystem.retrieve_sam("CECMod")
    cec_inverters_0 = pvlib.pvsystem.retrieve_sam("cecinverter")

    logger.info("Reading the downloaded database from SAM")
    cec_modules = pvlib.pvsystem.retrieve_sam(path=str(sam_dir / "CEC Modules.csv"))
    cec_modules = cec_modules.loc[:, ~cec_modules.columns.duplicated()]  # Drop column duplicates
    cec_inverters = pvlib.pvsystem.retrieve_sam(path=str(sam_dir / "CEC Inverters.csv"))
    cec_inverters = cec_inverters.loc[
        :, ~cec_inverters.columns.duplicated()
    ]  # Drop column duplicates

    logger.info("Reading custom EMHASS database")
    cec_modules_emhass = pvlib.pvsystem.retrieve_sam(path=str(data_path / "emhass_modules.csv"))
    cec_inverters_emhass = pvlib.pvsystem.retrieve_sam(path=str(data_path / "emhass_inverters.csv"))
    logger.info(strait_str)
    logger.info(strait_str)

    logger.info("Updating databases")

    # Modules
    cec_modules, cols = append_missing_columns(cec_modules, cec_modules_0)
    logger.info(
        f"Number of elements from old database copied in new database for modules = {len(cols)}:"
    )
    cec_modules, cols = append_missing_columns(cec_modules, cec_modules_emhass)
    logger.info(
        "Number of elements from custom EMHASS database copied in new database "
        f"for modules = {len(cols)}:"
    )

    # Inverters
    cec_inverters, cols = append_missing_columns(cec_inverters, cec_inverters_0)
    logger.info(
        f"Number of elements from old database copied in new database for inverters = {len(cols)}"
    )
    cec_inverters, cols = append_missing_columns(cec_inverters, cec_inverters_emhass)
    logger.info(
        "Number of elements from custom EMHASS database copied in new database "
        f"for inverters = {len(cols)}"
    )
    logger.info(strait_str)
    logger.info(strait_str)
    return {"cec_modules": plain_labels(cec_modules), "cec_inverters": plain_labels(cec_inverters)}


def tables_match(committed: pd.DataFrame, rebuilt: pd.DataFrame) -> bool:
    """Same labels, values and value types (DataFrame.equals alone takes 60 and 60.0 as equal)."""
    return (
        committed.equals(rebuilt)
        and committed.index.dtype == rebuilt.index.dtype
        and committed.columns.dtype == rebuilt.columns.dtype
        and committed.map(type).equals(rebuilt.map(type))
    )


def describe_differences(name: str, committed: pd.DataFrame, rebuilt: pd.DataFrame) -> list[str]:
    """Return readable lines describing how rebuilt differs from committed (empty if equal)."""
    if tables_match(committed, rebuilt):
        return []
    lines = [f"{name}: committed {committed.shape} vs rebuilt {rebuilt.shape} (rows x models)"]
    if (committed.index.dtype, committed.columns.dtype) != (
        rebuilt.index.dtype,
        rebuilt.columns.dtype,
    ):
        lines.append(
            "  the row or column labels have another dtype: committed "
            f"{committed.index.dtype}/{committed.columns.dtype}, "
            f"rebuilt {rebuilt.index.dtype}/{rebuilt.columns.dtype}"
        )
        # Compare the rest with plain labels on both sides.
        committed, rebuilt = plain_labels(committed), plain_labels(rebuilt)
    labels_line = len(lines)
    added = [c for c in rebuilt.columns if c not in committed.columns]
    removed = [c for c in committed.columns if c not in rebuilt.columns]
    if added:
        lines.append(f"  {len(added)} model(s) only in rebuilt, e.g. {added[:5]}")
    if removed:
        lines.append(f"  {len(removed)} model(s) only in committed, e.g. {removed[:5]}")
    if not committed.index.equals(rebuilt.index):
        rows_added = [r for r in rebuilt.index if r not in committed.index]
        rows_removed = [r for r in committed.index if r not in rebuilt.index]
        lines.append(
            f"  parameter rows differ: only in rebuilt {rows_added}, only in committed {rows_removed}"
            + ("" if rows_added or rows_removed else ", same rows in a different order")
        )
    else:
        changed = [
            c
            for c in committed.columns
            if c in rebuilt.columns and not committed[c].equals(rebuilt[c])
        ]
        if changed:
            lines.append(f"  {len(changed)} model(s) with changed values, e.g. {changed[:5]}")
    common = [c for c in committed.columns if c in rebuilt.columns]
    if not added and not removed and common != list(rebuilt.columns):
        lines.append("  same models but in a different column order")
    if len(lines) == labels_line and committed.equals(rebuilt):
        retyped = [
            c for c in committed.columns if not committed[c].map(type).equals(rebuilt[c].map(type))
        ]
        if retyped:
            lines.append(
                f"  {len(retyped)} model(s) with the same values but other types "
                f"(for example 60 and 60.0), e.g. {retyped[:5]}"
            )
    if len(lines) == 1:
        lines.append("  tables differ in dtypes or layout")
    return lines


def check_databases(sam_dir: pathlib.Path) -> int:
    """Rebuild in memory and compare with the committed files. Returns a process exit code."""
    rebuilt = build_databases(sam_dir)
    problems = []
    for name in database_names:
        path = data_path / f"{name}.pbz2"
        try:
            committed = read_committed_database(path)
        except READ_ERRORS as exc:
            problems.append(f"{name}: cannot read {path.name}: {exc}")
            continue
        diff = describe_differences(name, committed, rebuilt[name])
        if diff:
            problems.extend(diff)
        else:
            logger.info(f"{name}: committed database matches the rebuild ({committed.shape})")
    if problems:
        for line in problems:
            logger.error(line)
        recorded = read_source().get("pandas_version")
        if recorded and recorded != pd.__version__:
            logger.error(
                f"pandas {pd.__version__} is installed but the databases were recorded with "
                f"pandas {recorded}; pandas versions can parse some floats differently (2 and "
                "3 do), so rebuild with the recorded version before reading anything into "
                "the differences."
            )
        logger.error(
            "The committed databases do not match a rebuild from the recorded sources. "
            "Rebuild them with this script from the SAM commit in "
            "scripts/cec_database_source.json, or update that file if the sources changed."
        )
        return 1
    return 0


def save_databases(databases: dict[str, pd.DataFrame]) -> list[str]:
    """Write each database whose content changed. Returns the names written.

    An unchanged table is left alone: pickling the same table with another pandas
    version gives different bytes, and rewriting it would only add a binary diff.
    """
    from tabulate import tabulate

    written = []
    for name, table in databases.items():
        path = data_path / f"{name}.pbz2"
        logger.info(f"{name} database")
        print(tabulate(table.head(20).iloc[:, :5], headers="keys", tablefmt="psql"))
        try:
            unchanged = tables_match(read_committed_database(path), table)
        except READ_ERRORS:
            unchanged = False
        if unchanged:
            logger.info(f"{name}: content unchanged, keeping {path.name}")
            continue
        data = cPickle.dumps(table)
        # The check has to be able to read back what is written here.
        if not tables_match(table, read_table(data, name)):
            raise RuntimeError(f"{name}: the pickled table does not read back unchanged")
        with bz2.BZ2File(path, "w") as f:
            f.write(data)
        written.append(name)
    return written


def read_source() -> dict:
    """The recorded sources, or an empty dict when there is no readable source file."""
    try:
        return json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def record_source(sam_commit: str) -> None:
    """Record the SAM commit and the pandas and pvlib versions the databases were built with."""
    source = read_source()
    source.update(
        {
            "sam_commit": sam_commit,
            "pandas_version": pd.__version__,
            "pvlib_version": pvlib.__version__,
        }
    )
    with open(source_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(source, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the CEC module and inverter databases used by EMHASS, "
        "or check the committed ones against a rebuild."
    )
    parser.add_argument(
        "--sam-dir",
        type=pathlib.Path,
        default=data_path,
        help='folder holding "CEC Modules.csv" and "CEC Inverters.csv" from SAM',
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare a rebuild with the committed databases instead of writing them",
    )
    parser.add_argument(
        "--sam-commit",
        help="when a database is rewritten, record this SAM commit in "
        "scripts/cec_database_source.json",
    )
    args = parser.parse_args(argv)
    if args.sam_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", args.sam_commit):
        parser.error("--sam-commit must be a full 40-character commit sha")
    if args.check:
        return check_databases(args.sam_dir)
    written = save_databases(build_databases(args.sam_dir))
    if args.sam_commit and (written or not source_path.exists()):
        record_source(args.sam_commit)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    sys.exit(main())
