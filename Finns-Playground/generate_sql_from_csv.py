#!/usr/bin/env python3
import csv, re, sys
from pathlib import Path

def sanitize_identifier(name: str) -> str:
    s = re.sub(r"\W+", "_", name).strip("_")
    if re.match(r"^\d", s): s = "_" + s
    return s or "col"

def infer_type(values):
    has_real = False
    for v in values:
        if v in (None, ""): continue
        vs = v.replace(",", ".") if v.count(",")==1 and v.count(".")==0 else v
        if re.fullmatch(r"[+-]?\d+", vs): continue
        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+\.)", vs): has_real = True; continue
        return "TEXT"
    return "REAL" if has_real else "INTEGER"

def escape_sql_value(val: str):
    if val in (None, ""): return "NULL"
    vs = val
    if re.fullmatch(r"[+-]?\d+", vs): return vs
    vs_norm = vs.replace(",", ".") if vs.count(",")==1 and vs.count(".")==0 else vs
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+\.)", vs_norm): return vs_norm
    return "'" + val.replace("'", "''") + "'"

def main(csv_path, table_name=None, out_path=None):
    csv_path = Path(csv_path)
    if not csv_path.exists(): raise SystemExit(f"CSV not found: {csv_path}")
    if table_name is None: table_name = sanitize_identifier(csv_path.stem)
    if out_path is None: out_path = csv_path.with_suffix(".sql")

    import csv
    with csv_path.open(newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows: raise SystemExit("CSV is empty.")

    header, data_rows = rows[0], rows[1:]
    # Sanitize & dedupe column names
    sanitized, seen = [], {}
    for col in header:
        s = sanitize_identifier(col)
        base = s or "col"
        i = seen.get(base, 0)
        if i: s = f"{base}_{i+1}"; seen[base] = i + 1
        else: seen[base] = 1
        sanitized.append(s)

    columns = list(zip(*data_rows)) if data_rows else [[] for _ in sanitized]
    types = [infer_type(col_values) for col_values in columns]

    create_lines = [f'CREATE TABLE "{table_name}" (']
    for name, typ in zip(sanitized, types):
        sql_type = "INTEGER" if typ=="INTEGER" else "REAL" if typ=="REAL" else "TEXT"
        create_lines.append(f'  "{name}" {sql_type},')
    create_lines[-1] = create_lines[-1].rstrip(",")
    create_lines.append(");")
    create_stmt = "\n".join(create_lines)

    col_list = ", ".join([f'"{c}"' for c in sanitized])
    values_chunks = []
    for r in data_rows:
        rr = list(r) + [None]*(len(sanitized)-len(r))
        rr = rr[:len(sanitized)]
        values = ", ".join(escape_sql_value(v) for v in rr)
        values_chunks.append(f"({values})")
    insert_stmt = f'INSERT INTO "{table_name}" ({col_list}) VALUES\n' + ",\n".join(values_chunks) + ";"

    sql_text = "-- Auto-generated from CSV\nPRAGMA foreign_keys = OFF;\nBEGIN TRANSACTION;\n" + create_stmt + "\n" + insert_stmt + "\nCOMMIT;\n"
    Path(out_path).write_text(sql_text, encoding="utf-8")
    print(f"Wrote SQL to {out_path}")

if __name__ == "__main__":
    # Usage: python generate_sql_from_csv.py input.csv [table_name] [out.sql]
    args = sys.argv[1:]
    if not args:
        print("Usage: python generate_sql_from_csv.py input.csv [table_name] [out.sql]")
        raise SystemExit(2)
    csv_path = args[0]
    table_name = args[1] if len(args) > 1 else None
    out_path = args[2] if len(args) > 2 else None
    main(csv_path, table_name, out_path)
