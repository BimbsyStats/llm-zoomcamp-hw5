import json
import dlt

# Load local agent traces dataset
with open("agent_traces.json", "r") as f:
    data = [json.loads(line) for line in f if line.strip()]

# Configure pipeline targeting agent_traces dataset
pipeline = dlt.pipeline(
    pipeline_name="agent_traces_pipeline",
    destination="duckdb",
    dataset_name="agent_traces"
)

# Run ingestion
info = pipeline.run(data, table_name="traces")
print(info)




