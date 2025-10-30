#!/usr/bin/env python3
import csv, re
from pathlib import Path

# === CONFIG (edit these four) ===
CSV_PATH = Path(r"./metacritic_games.csv")
TABLE_NAME = "Steam"
OUTPUT_SQL = Path(r"./Games.sql")
HAS_HEADER = True
BATCH_SIZE = 10000
# ================================


def to_identifier(index):
    return f"col{index+1}"


def is_int(s: str) -> bool:
    import re

    return re.fullmatch(r"[+-]?\d+", s) is not None


def is_float(s: str) -> bool:
    import re

    return re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+\.)", s) is not None


def format_value(val: str) -> str:
    if val is None or val == "":
        return "NULL"
    v = val.replace(",", ".") if (val.count(",") == 1 and val.count(".") == 0) else val
    if is_int(v) or is_float(v):
        return v
    escaped = val.replace('"', '\\"')
    return f'"' + escaped + '"'


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit("CSV is empty.")
    header = rows[0] if HAS_HEADER else None
    data = rows[1:] if HAS_HEADER else rows
    return header, data


def build_create_table(n_cols: int) -> str:
    cols = [f"{to_identifier(i)} TEXT" for i in range(n_cols)]
    return f"CREATE TABLE {TABLE_NAME} (\n  " + ",\n  ".join(cols) + "\n);"


def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def build_insert_batches(data_rows):
    statements = []
    for i in range(0, len(data_rows), BATCH_SIZE):
        batch = data_rows[i : i + BATCH_SIZE]
        values_lines = []
        for r in batch:
            vals = [format_value(v) for v in r]
            values_lines.append("(" + ", ".join(vals) + ")")
        statements.append(
            f"INSERT INTO {TABLE_NAME} VALUES\n  " + ",\n  ".join(values_lines) + "\n;"
        )
    return statements


def main():
    header, data_rows = read_csv_rows(CSV_PATH)
    num_cols = (
        max(len(r) for r in data_rows) if data_rows else (len(header) if header else 0)
    )
    norm_rows = [(row + [""] * (num_cols - len(row)))[:num_cols] for row in data_rows]

    create_stmt = build_create_table(num_cols)
    insert_stmts = build_insert_batches(norm_rows)

    sql_text = create_stmt + "\n\n" + "\n\n".join(insert_stmts) + "\n"
    OUTPUT_SQL.write_text(sql_text, encoding="utf-8")
    print(f"Wrote SQL to {OUTPUT_SQL}")


if __name__ == "__main__":
    main()
