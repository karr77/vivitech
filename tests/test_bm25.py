import json
import re
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType
from collections import defaultdict
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

# --- 1. Configuration ---
MILVUS_URI = "http://localhost:19530"
# In a real-world scenario, use a secure way to manage tokens, e.g., environment variables
MILVUS_TOKEN = "root:Milvus"
# Use a new collection name to avoid conflicts with the old schema
COLLECTION_NAME = "xianyu_items_intelligent_hybrid"
DATA_PATH = "data/full_item_detailes.json"

# We use a powerful open-source Chinese embedding model.
# Using a GPU for the model will significantly speed up embedding generation.
# bge-small-zh-v1.5 is a good balance between performance and resource consumption.
EMBEDDING_MODEL = 'BAAI/bge-small-zh-v1.5'
DIMENSION = 512  # Dimension of the bge-small-zh-v1.5 model

# --- Advanced Model Extractor (based on user's example) ---
@dataclass
class ModelCandidate:
    text: str
    confidence: float
    method: str
    context: str


class AdvancedModelExtractor:
    """
    A multi-strategy model extractor.
    For this implementation, we focus on the rule-based method as provided,
    with placeholders for more advanced techniques like vector or transformer models.
    """
    def __init__(self):
        # In a real-world application, these would be fully implemented models.
        # self.vector_matcher = VectorModelMatcher()
        # self.transformer_extractor = TransformerModelExtractor()
        self.confidence_threshold = 0.6
        
    async def extract_models(self, text: str) -> List[ModelCandidate]:
        """
        Extracts model candidates from text using a rule-based approach.
        """
        candidates = []
        
        # We are using only the rule-based method for this example.
        rule_candidates = self._rule_based_extract(text)
        for candidate in rule_candidates:
            candidates.append(ModelCandidate(
                text=candidate,
                confidence=0.7, # Assign a base confidence for rule-based findings
                method="rule",
                context=self._get_context(text, candidate)
            ))
        
        # The ensemble logic is kept for future expansion with more methods.
        return self._ensemble_results(candidates)
    
    def _rule_based_extract(self, text: str) -> List[str]:
        """
        A set of robust regex patterns to find potential model numbers,
        combined with contextual validation.
        """
        patterns = [
            # More specific: "型号: XXX", "model: XXX", etc.
            r'(?:型号|model|编号|款号|货号)[\s:：]*([a-zA-Z0-9\-\/]{3,20})',
            # More generic: Captures mixed alpha-numeric codes like '4QUE008' or 'K20-Pro'
            # It looks for sequences with letters and numbers.
            r'\b([a-zA-Z\d]{2,}[-]?\d{2,}[a-zA-Z\d]*|[a-zA-Z]{2,}[-]?\d{2,}[a-zA-Z\d]*)\b',
            # Catches codes followed by common product type keywords
            r'\b([A-Z0-9]{3,15})\b(?=\s*(?:型|款|版|背包|书包))',
        ]
        
        candidates = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Filter out pure numbers and very short matches
                if len(match) > 3 and not match.isdigit():
                     candidates.add(match)
        
        # --- Contextual Validation Step ---
        # List of keywords that increase the likelihood of a candidate being a model number.
        context_keywords = ['型号', '款', '货号', '编号', 'model', '有']
        
        validated_candidates = []
        for candidate in candidates:
            # Check for keywords in the vicinity of the candidate
            try:
                # Case-insensitive search
                start_index = text.lower().find(candidate.lower())
                # Define a window of 10 chars before and after the candidate
                context_start = max(0, start_index - 10)
                context_end = min(len(text), start_index + len(candidate) + 10)
                context_window = text[context_start:context_end]
                
                # If any keyword is found in the context window, it's a strong signal.
                if any(keyword in context_window for keyword in context_keywords):
                    validated_candidates.append(candidate)
            except:
                # In case of errors, we can be lenient and just add the candidate
                validated_candidates.append(candidate)

        # If contextual validation yields results, we prefer them.
        # Otherwise, we fall back to the initial regex-based candidates.
        return validated_candidates if validated_candidates else list(candidates)
    
    def _ensemble_results(self, candidates: List[ModelCandidate]) -> List[ModelCandidate]:
        """
        A simple result fusion logic. Currently handles single-method results.
        Designed to be extended for multiple extraction methods.
        """
        if not candidates:
            return []

        # Group candidates by their text to handle duplicates
        grouped = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.text.upper()].append(candidate)
        
        final_results = []
        for text_upper, group in grouped.items():
            # For now, we take the first candidate found for a given text.
            # Confidence fusion logic can be added here if multiple methods are used.
            candidate = group[0]
            if candidate.confidence >= self.confidence_threshold:
                final_results.append(ModelCandidate(
                    text=candidate.text,
                    confidence=candidate.confidence,
                    method=candidate.method,
                    context=candidate.context
                ))
        
        return sorted(final_results, key=lambda x: x.confidence, reverse=True)
    
    def _get_context(self, text: str, model: str) -> str:
        """
        Extracts a snippet of text around the found model for context.
        """
        try:
            # Case-insensitive search for the model
            start = text.lower().find(model.lower())
            if start == -1:
                return f"'{model}' not found in text for context."
            
            # Get 15 characters of context before and after the model
            context_start = max(0, start - 15)
            context_end = min(len(text), start + len(model) + 15)
            return f"...{text[context_start:context_end]}..."
        except Exception as e:
            return f"Error getting context: {e}"

# --- 2. Milvus Setup and Data Ingestion ---
def setup_milvus_collection():
    """
    Connects to Milvus, sets up a simplified collection schema (no BM25),
    and ingests data from the JSON file.
    """
    print("--- Setting up Milvus collection (Intelligent Hybrid) ---")
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    if client.has_collection(collection_name=COLLECTION_NAME):
        print(f"Dropping existing collection: {COLLECTION_NAME}")
        client.drop_collection(collection_name=COLLECTION_NAME)

    # A simplified schema without sparse vectors for BM25
    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=20)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=500)
    # This field stores the text used for generating embeddings
    schema.add_field(field_name="text_for_embedding", datatype=DataType.VARCHAR, max_length=4000)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)

    # We only need an index for the dense vector field
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="L2", # L2 is a common choice for BGE models
        params={"M": 16, "efConstruction": 256}
    )

    print(f"Creating collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )

    # --- Data Preparation and Ingestion ---
    print("--- Preparing and ingesting data ---")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    data_to_insert = []
    for item in data:
        title = item.get('title', '')
        description = item.get('full_description', '')
        # The text for embedding combines title and description for rich semantics.
        text_for_embedding = f"{title} {description}"

        data_to_insert.append({
            "id": item['id'],
            "title": item['title'],
            "text_for_embedding": text_for_embedding,
        })

    # Generate dense embeddings for all texts in a batch
    print("Generating dense embeddings...")
    texts_to_embed = [item["text_for_embedding"] for item in data_to_insert]
    dense_embeddings = model.encode(texts_to_embed, show_progress_bar=True)

    # Add the generated embeddings to our data
    for i, item in enumerate(data_to_insert):
        item['dense_vector'] = dense_embeddings[i]

    print(f"Inserting {len(data_to_insert)} records into Milvus...")
    client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    
    print("Loading collection into memory for searching...")
    client.load_collection(collection_name=COLLECTION_NAME)

    print("--- Setup and data ingestion complete ---")


def reciprocal_rank_fusion(search_results_list, weights, k=60):
    """
    Performs Reciprocal Rank Fusion on a list of search results.
    `k` is a constant used in the RRF formula to control the influence of lower ranks.
    """
    fused_scores = defaultdict(float)
    
    if len(search_results_list) != len(weights):
        raise ValueError("Length of search_results_list and weights must be the same.")

    # Iterate through each list of search results
    for i, results in enumerate(search_results_list):
        if not results or not results[0]:
            continue
        # For each document in the results, update its score
        for rank, result in enumerate(results[0]):
            doc_id = result.entity.get('id')
            # The RRF formula: score += weight * (1 / (k + rank))
            fused_scores[doc_id] += weights[i] * (1 / (k + rank))
            
    # Sort the documents by their final fused scores in descending order
    reranked_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    
    return reranked_results


# --- 3. Intelligent Hybrid Search Implementation ---
async def hybrid_search(query: str, top_k: int = 5):
    """
    Performs an intelligent hybrid search:
    1. Extracts model numbers from the query using AdvancedModelExtractor.
    2. Conducts a semantic search on the original query.
    3. If models are found, conducts a separate, targeted search for them.
    4. Fuses the results using RRF, giving more weight to model matches.
    """
    print(f"\n--- Performing intelligent hybrid search for query: '{query}' ---")
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    model_extractor = AdvancedModelExtractor()

    # 1. Extract model numbers from the query
    print("--- Extracting model candidates from query... ---")
    extracted_models = await model_extractor.extract_models(query)
    if extracted_models:
        print("--- Extracted Model Candidates ---")
        for model in extracted_models:
            print(f"  - Text: '{model.text}', Confidence: {model.confidence:.2f}, Context: {model.context}")
    else:
        print("--- No valid model candidates extracted from query. ---")

    # 2. Semantic Search (on the original query)
    print("Executing semantic search...")
    query_embedding = embedding_model.encode(query)
    semantic_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        anns_field="dense_vector",
        limit=top_k,
        output_fields=["id", "title"]
    )

    # 3. Targeted Model Search (if any models were extracted)
    model_search_results = []
    if extracted_models:
        print("Executing targeted search for extracted models...")
        # Create a query string with just the model numbers
        model_query = " ".join([m.text for m in extracted_models])
        model_query_embedding = embedding_model.encode(model_query)
        
        model_search_results = client.search(
            collection_name=COLLECTION_NAME,
            data=[model_query_embedding],
            anns_field="dense_vector",
            limit=top_k,
            output_fields=["id", "title"]
        )

    # 4. Fuse results using RRF
    print("Fusing search results...")
    # Give more weight to the targeted model search if it was performed
    if extracted_models:
        search_lists = [semantic_results, model_search_results]
        weights = [0.4, 0.6] # Emphasize model matches
    else:
        search_lists = [semantic_results]
        weights = [1.0] # Only semantic results are available

    reranked_ids_with_scores = reciprocal_rank_fusion(search_lists, weights=weights)

    # 5. Fetch and display the top_k reranked results
    print("\n--- Fused and Reranked Search Results ---")
    if not reranked_ids_with_scores:
        print("No results found.")
        return

    # Get the IDs of the top results
    top_ids = [item[0] for item in reranked_ids_with_scores[:top_k]]
    if not top_ids:
        print("No results found after fusion.")
        return

    # Fetch the full documents for these top results
    final_results = client.get(collection_name=COLLECTION_NAME, ids=top_ids, output_fields=["id", "title"])
    
    # Create a map to preserve the new reranked order
    id_to_doc = {item['id']: item for item in final_results}

    for i, (doc_id, score) in enumerate(reranked_ids_with_scores[:top_k]):
        doc = id_to_doc.get(doc_id)
        if doc:
            print(f"Rank {i + 1}:")
            print(f"  ID: {doc.get('id')}")
            print(f"  Title: {doc.get('title')}")
            print(f"  RRF Score: {score:.6f}")
            print("-" * 20)


# --- 4. Main Execution Block ---
async def main():
    # This function will set up the Milvus collection and ingest data.
    # It only needs to be run once. After the first successful execution,
    # you can comment it out to speed up subsequent test runs.
    setup_milvus_collection()

    # --- Test Queries ---
    
    # Query with a clear model number
    await hybrid_search(query="我看你主页的包好多型号，有4QUE008 这个型号吗")

    # A more semantic query without a specific model number
    await hybrid_search(query="适合户外活动的防水背包，比如始祖鸟那种")
    
    # Query with a model number that might be in the description but not title
    await hybrid_search(query="有没有FYA-213这个款")


if __name__ == "__main__":
    # Use asyncio.run() to execute the async main function
    asyncio.run(main())
