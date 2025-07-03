import json
from tqdm import tqdm
from openai import OpenAI
from memory_layer import AgenticMemorySystem
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.text_splitter import RecursiveCharacterTextSplitter

base_url = "http://123.129.219.111:3000/v1"
api_key = "sk-KVBaSbapLJHypzMgqurxaRYaCKDMR8CDGlTQQ5VpD74fBbzE"
client = OpenAI(api_key=api_key, base_url=base_url)

def analyze_content(content, visual, model="gpt-4o-mini"):
    """Analyze content to extract keywords, context, and other metadata"""

    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                            },
                            "required": ["keywords", "context", "tags"],
                            "additionalProperties": False
                        },
                        "strict": True
                }
            }

    if not visual:
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content

        try:
            response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=0.7,
            max_tokens=2048)
            
            analysis = json.loads(response.choices[0].message.content)

            return analysis
            
        except Exception as e:
            print(f"Error analyzing content: {str(e)}")
            print(f"Raw response: {response}")
            return {
                "keywords": [],
                "context": "General",
                "category": "Uncategorized",
                "tags": []
            }

    else:
        prompt = """Generate a structured analysis of the provided image (which is a page of a scientific paper) by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // A summary of:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                    // - Detailed descriptions of visual elements within the image, such as tables and charts (including the index, such as Table 1, Figure 1, etc.)
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }
            """

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": [{"type":"text","text": prompt},{"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{content}"}}]},
            ],
            response_format=response_format,
            temperature=0.7,
            max_tokens=2048
            )
            analysis = json.loads(response.choices[0].message.content)

            return analysis
            
        except Exception as e:
            print(f"Error analyzing content: {str(e)}")
            print(f"Raw response: {response}")
            return {
                "keywords": [],
                "context": "General",
                "category": "Uncategorized",
                "tags": []
            }


def embedder(documents, model="text-embedding-3-small"):
    # generate embeddings in parallel using multithreading
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(client.embeddings.create, model=model, input=doc) for doc in documents]
        responses = [future.result() for future in futures]
    return [response.data[0].embedding for response in responses]


def construct_memory(context, chunk_size, model_name = "gpt-4o-mini"):
    
    # initialize memory system
    print("initializing memory system")
    memory_system = AgenticMemorySystem(
        llm_backend="openai", llm_model=model_name, 
        api_key="sk-KVBaSbapLJHypzMgqurxaRYaCKDMR8CDGlTQQ5VpD74fBbzE")
    
    # chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=300)
    chunks = text_splitter.split_text(context)
    
    # analyze chunks 
    text_contents = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(analyze_content, chunk, False, model_name) for chunk in chunks]
        results = list(tqdm(as_completed(futures), total=len(futures), desc="Analyzing content"))
        for i, result in enumerate(results):
            text_contents[i] = result.result()

    # embed chunks
    documents = [text_contents[i]["context"] + " keywords: " + ", ".join(text_contents[i]["keywords"]) for i in range(len(text_contents))]
    text_embeddings = embedder(documents)
    for i, embedding in enumerate(text_embeddings):
        text_contents[i]["embedding"] = [embedding]

    # add chunks to memory system
    for i, chunk in enumerate(chunks):
        memory_system.add_note(content = chunk, context = text_contents[i]["context"], 
                                keywords = text_contents[i]["keywords"],
                                tags = text_contents[i]["tags"], 
                                category = "Uncategorized",
                                pre_embeddings = text_contents[i]["embedding"])
    return memory_system


def search_memory(memory_system, query, k=10):
    notes = memory_system.find_related_notes(query, k=k)
    results = [note.content for note in notes]
    return results

