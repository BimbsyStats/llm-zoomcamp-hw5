import duckdb

con = duckdb.connect("from_database.duckdb")

# Show all schemas and tables in the database
tables = con.sql(
    "SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema NOT IN ('information_schema', 'pg_catalog');"
).fetchall()

print(f"Total tables found: {len(tables)}\n")
for schema, table in tables:
    print(f"Schema: {schema} | Table: {table}")


