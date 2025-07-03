import json
from amem_utils import construct_memory, search_memory

with open("/home/ubuntu/songjunru/long_context/longbench_single_QA_data.json", "r") as f:
    data = json.load(f)

context = data[0]["context"]
memory_system = construct_memory(context = context, chunk_size = 2048, model_name = "gpt-4o-mini")
results = search_memory(memory_system = memory_system, query = "This is an example query.", k = 10)
print(results)



