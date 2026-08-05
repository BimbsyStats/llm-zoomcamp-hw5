import argparse
import glob
import duckdb
import os


def scan_duckdb_files(recursive: bool):
    pattern = "**/*.duckdb" if recursive else "*.duckdb"
    return glob.glob(pattern, recursive=recursive)


def get_tables_from_db(db_file: str):
    with duckdb.connect(db_file) as con:
        rows = con.sql(
            "SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema NOT IN ('information_schema', 'pg_catalog');"
        ).fetchall()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Scan .duckdb files and count tables.")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories for .duckdb files")
    parser.add_argument("--expected", type=int, default=24, help="Expected total table count (default: 24)")
    args = parser.parse_args()

    print("--- SCANNING ALL DUCKDB FILES IN WORKSPACE ---\n")

    duckdb_files = scan_duckdb_files(args.recursive)

    all_tables = []

    if not duckdb_files:
        print("No .duckdb files found in workspace.")
        return

    for db_file in duckdb_files:
        try:
            tables = get_tables_from_db(db_file)
            print(f"📁 Database File: {db_file}")
            print(f"   Total Tables: {len(tables)}")

            # Count by schema
            schemas = set(t[0] for t in tables)
            for s in schemas:
                schema_count = sum(1 for t in tables if t[0] == s)
                print(f"   └── Schema '{s}': {schema_count} tables")

            # record full table identifiers
            for s, t in tables:
                all_tables.append(f"{s}.{t}")

            print("-" * 45)
        except Exception as e:
            print(f"Error checking {db_file}: {e}")

    total_tables = len(all_tables)
    print(f"\nSummary: Found {total_tables} total tables across {len(duckdb_files)} database file(s).")
    if total_tables == args.expected:
        print(f"✅ Expected table count {args.expected} found.")
    else:
        print(f"⚠️ Expected {args.expected} tables but found {total_tables}.")
        # show table list to help debug
        for t in sorted(all_tables):
            print(f" - {t}")


if __name__ == "__main__":
    main()


