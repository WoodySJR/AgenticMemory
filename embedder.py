from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

base_url = "http://123.129.219.111:3000/v1"
api_key = "sk-KVBaSbapLJHypzMgqurxaRYaCKDMR8CDGlTQQ5VpD74fBbzE"
client = OpenAI(api_key=api_key, base_url=base_url)

def embedder(documents, model="text-embedding-3-small"):
    # generate embeddings in parallel using multithreading
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(client.embeddings.create, model=model, input=doc) for doc in documents]
        responses = [future.result() for future in futures]
    return [response.data[0].embedding for response in responses]