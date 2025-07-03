import json
from amem_utils import construct_memory, search_memory

with open("/home/ubuntu/songjunru/long_context/longbench_single_QA_data.json", "r") as f:
    data = json.load(f)

context = data[0]["context"]
memory_system = construct_memory(context, 2048)
results = search_memory(memory_system, "This is an example query.", k=10)
print(results)



