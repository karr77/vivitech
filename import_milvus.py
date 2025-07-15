import json
import os
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer

# Milvus 连接参数
MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
MILVUS_SECURE = False

# 数据库名称
DATABASE_NAME = "test_data"

# 集合名称
COLLECTION_NAME = "products"

# 向量维度
VECTOR_DIM = 1024  # 使用 BAAI/bge-large-zh-v1.5 模型，它生成1024维向量

def connect_to_milvus():
    """连接到Milvus服务器的指定数据库"""
    try:
        connections.connect(
            alias="default",  # 给连接一个别名
            uri=MILVUS_URI,
            token=MILVUS_TOKEN,
            secure=MILVUS_SECURE,
            db_name=DATABASE_NAME  # 直接连接到test_data数据库
        )
        print(f"成功连接到 Milvus 服务器 {MILVUS_URI} 的 {DATABASE_NAME} 数据库")
        return True
    except Exception as e:
        print(f"连接 Milvus 数据库 {DATABASE_NAME} 失败: {e}")
        return False

def create_collection():
    """创建集合，如果不存在"""
    if utility.has_collection(COLLECTION_NAME):
        print(f"集合 '{COLLECTION_NAME}' 已存在")
        return Collection(COLLECTION_NAME)
    
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="brand", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="model", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="product_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="color", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="size", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="capacity", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="price", dtype=DataType.FLOAT),
        FieldSchema(name="embedding_text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="embedding_vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    ]
    
    # 定义集合模式
    schema = CollectionSchema(fields, "产品集合")
    
    # 创建集合
    collection = Collection(COLLECTION_NAME, schema)
    
    # 为向量创建索引
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index("embedding_vector", index_params)
    print(f"成功创建集合 '{COLLECTION_NAME}' 并添加索引")
    
    return collection

def load_products_from_json(file_path):
    """从JSON文件加载产品数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            products = json.load(f)
        print(f"从 {file_path} 成功加载了 {len(products)} 条产品记录")
        return products
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        return []

def generate_embeddings(texts, model):
    """使用句子转换器生成文本嵌入"""
    try:
        embeddings = model.encode(texts)
        return embeddings
    except Exception as e:
        print(f"生成嵌入失败: {e}")
        return None

def insert_data(collection, products, model):
    """将产品数据插入到Milvus集合中"""
    # 准备数据
    ids = [product["id"] for product in products]
    brands = [product.get("brand", "") for product in products]
    models = [product.get("model", "") for product in products]
    product_types = [product.get("product_type", "") for product in products]
    colors = [product.get("color", "") for product in products]
    sizes = [product.get("size", "") for product in products]
    capacities = [product.get("capacity", "") for product in products]
    prices = [product.get("price", 0.0) for product in products]
    embedding_texts = [product.get("embedding_text", "") for product in products]
    
    # 生成向量嵌入
    print("正在为产品文本生成嵌入向量...")
    embedding_vectors = generate_embeddings(embedding_texts, model)
    
    if embedding_vectors is None:
        print("生成嵌入向量失败，无法继续")
        return
    
    # 插入数据
    entities = [
        ids, 
        brands, 
        models, 
        product_types, 
        colors, 
        sizes, 
        capacities, 
        prices, 
        embedding_texts, 
        embedding_vectors
    ]
    
    try:
        collection.insert(entities)
        collection.flush()
        print(f"成功插入 {len(ids)} 条记录到集合 '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"插入数据失败: {e}")

def main():
    # 连接到Milvus的test_data数据库
    if not connect_to_milvus():
        print("尝试连接到默认数据库...")
        try:
            connections.connect(
                alias="default",
                uri=MILVUS_URI,
                token=MILVUS_TOKEN,
                secure=MILVUS_SECURE
            )
            print("已连接到默认数据库")
        except Exception as e:
            print(f"连接到默认数据库也失败: {e}")
            return
    
    # 创建集合
    collection = create_collection()
    
    # 加载产品数据
    products = load_products_from_json("data/cleaned_product.json")
    if not products:
        return
    
    # 加载嵌入模型
    try:
        print("加载文本嵌入模型...")
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    except Exception as e:
        print(f"加载嵌入模型失败: {e}")
        return
    
    # 插入数据到集合
    insert_data(collection, products, model)
    
    # 显示集合统计信息
    print(f"集合 '{COLLECTION_NAME}' 统计信息:")
    print(f"实体数量: {collection.num_entities}")
    print("数据导入完成!")

if __name__ == "__main__":
    main()
