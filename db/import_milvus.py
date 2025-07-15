import json
import numpy as np
import os
from typing import List, Dict, Any
from pymilvus import (
    connections, 
    Collection, 
    CollectionSchema, 
    FieldSchema, 
    DataType,
    utility,
    db
)
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusDataImporter:
    """Milvus数据导入器"""
    
    def __init__(self, 
                 uri: str = "http://43.140.216.157:18001",
                 token: str = "root:Haipro003838.",
                 db_name: str = "test_data",
                 secure: bool = False):
        """
        初始化Milvus连接
        
        Args:
            uri: Milvus服务器地址
            token: 认证token
            db_name: 数据库名称
            secure: 是否使用安全连接
        """
        self.uri = uri
        self.token = token
        self.db_name = db_name
        self.secure = secure
        self.connection_alias = "default"
        
        # 集合名称
        self.product_collection_name = "xianyu_products"
        self.qa_collection_name = "xianyu_qa_pairs_pairs"
        
        # 向量维度（根据之前的embedding结果）
        self.vector_dim = 768
        
        # 连接到Milvus
        self.connect_to_milvus()

    def connect_to_milvus(self):
        """连接到Milvus数据库"""
        try:
            # 建立连接
            connections.connect(
                alias=self.connection_alias,
                uri=self.uri,
                token=self.token,
                secure=self.secure
            )
            
            # 检查并创建数据库
            if self.db_name not in db.list_database():
                db.create_database(self.db_name)
                logger.info(f"数据库 {self.db_name} 创建成功")
            
            # 使用指定数据库
            db.using_database(self.db_name)
            logger.info(f"成功连接到Milvus数据库: {self.db_name}")
            
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise

    def create_product_collection_schema(self) -> CollectionSchema:
        """创建商品集合的Schema"""
        fields = [
            # 主键字段
            FieldSchema(
                name="id", 
                dtype=DataType.VARCHAR, 
                max_length=100,
                is_primary=True, 
                auto_id=False,
                description="商品唯一ID"
            ),
            
            # 向量字段
            FieldSchema(
                name="embedding", 
                dtype=DataType.FLOAT_VECTOR, 
                dim=self.vector_dim,
                description="商品embedding向量"
            ),
            
            # 商品基本信息
            FieldSchema(
                name="title", 
                dtype=DataType.VARCHAR, 
                max_length=500,
                description="商品标题"
            ),
            FieldSchema(
                name="description", 
                dtype=DataType.VARCHAR, 
                max_length=2000,
                description="商品描述"
            ),
            FieldSchema(
                name="brand", 
                dtype=DataType.VARCHAR, 
                max_length=100,
                description="商品品牌"
            ),
            FieldSchema(
                name="category", 
                dtype=DataType.VARCHAR, 
                max_length=50,
                description="商品分类"
            ),
            FieldSchema(
                name="subcategory", 
                dtype=DataType.VARCHAR, 
                max_length=50,
                description="商品子分类"
            ),
            
            # 商品状态和价格
            FieldSchema(
                name="price", 
                dtype=DataType.DOUBLE,
                description="商品价格"
            ),
            FieldSchema(
                name="condition", 
                dtype=DataType.VARCHAR, 
                max_length=20,
                description="商品成色"
            ),
            FieldSchema(
                name="status", 
                dtype=DataType.VARCHAR, 
                max_length=20,
                description="商品状态(在售/已售出/下架)"
            ),
            
            # 地理和用户信息
            FieldSchema(
                name="location", 
                dtype=DataType.VARCHAR, 
                max_length=50,
                description="商品所在城市"
            ),
            FieldSchema(
                name="seller_id", 
                dtype=DataType.VARCHAR, 
                max_length=50,
                description="卖家ID"
            ),
            
            # 统计信息
            FieldSchema(
                name="views", 
                dtype=DataType.INT64,
                description="浏览次数"
            ),
            FieldSchema(
                name="likes", 
                dtype=DataType.INT64,
                description="喜欢次数"
            ),
            
            # 时间信息
            FieldSchema(
                name="created_at", 
                dtype=DataType.VARCHAR, 
                max_length=50,
                description="创建时间"
            ),
            
            # 用于embedding的文本
            FieldSchema(
                name="embedding_text", 
                dtype=DataType.VARCHAR, 
                max_length=2000,
                description="用于生成embedding的文本"
            )
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="闲鱼商品信息集合",
            enable_dynamic_field=True
        )
        
        return schema

    def create_qa_collection_schema(self) -> CollectionSchema:
        """创建问答集合的Schema"""
        fields = [
            # 主键字段
            FieldSchema(
                name="id", 
                dtype=DataType.VARCHAR, 
                max_length=100,
                is_primary=True, 
                auto_id=True,
                description="问答对唯一ID"
            ),
            
            # 问题向量字段
            FieldSchema(
                name="question_embedding", 
                dtype=DataType.FLOAT_VECTOR, 
                dim=self.vector_dim,
                description="问题embedding向量"
            ),
            
            # 答案向量字段
            FieldSchema(
                name="answer_embedding", 
                dtype=DataType.FLOAT_VECTOR, 
                dim=self.vector_dim,
                description="答案embedding向量"
            ),
            
            # 关联商品ID
            FieldSchema(
                name="product_id", 
                dtype=DataType.VARCHAR, 
                max_length=100,
                description="关联的商品ID"
            ),
            
            # 问答内容
            FieldSchema(
                name="question", 
                dtype=DataType.VARCHAR, 
                max_length=1000,
                description="用户问题"
            ),
            FieldSchema(
                name="answer", 
                dtype=DataType.VARCHAR, 
                max_length=1000,
                description="客服回答"
            ),
            
            # 问题分类
            FieldSchema(
                name="question_type", 
                dtype=DataType.VARCHAR, 
                max_length=50,
                description="问题类型"
            ),
            
            # 时间信息
            FieldSchema(
                name="timestamp", 
                dtype=DataType.VARCHAR, 
                max_length=50,
                description="问答时间"
            ),
            
            # 用于embedding的文本
            FieldSchema(
                name="question_embedding_text", 
                dtype=DataType.VARCHAR, 
                max_length=1000,
                description="用于生成问题embedding的文本"
            ),
            FieldSchema(
                name="answer_embedding_text", 
                dtype=DataType.VARCHAR, 
                max_length=1000,
                description="用于生成答案embedding的文本"
            )
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="闲鱼问答对集合",
            enable_dynamic_field=True
        )
        
        return schema

    def create_collection_if_not_exists(self, collection_name: str, schema: CollectionSchema) -> Collection:
        """检查并创建集合"""
        if utility.has_collection(collection_name):
            logger.info(f"集合 {collection_name} 已存在")
            collection = Collection(collection_name)
        else:
            logger.info(f"创建集合 {collection_name}")
            collection = Collection(
                name=collection_name,
                schema=schema,
                using=self.connection_alias
            )
            logger.info(f"集合 {collection_name} 创建成功")
        
        return collection

    def create_indexes(self, collection: Collection, is_qa_collection: bool = False):
        """为集合创建索引"""
        # 向量索引参数
        index_params = {
            "metric_type": "COSINE",  # 使用余弦相似度
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        
        if is_qa_collection:
            # 为问答集合创建两个向量字段的索引
            vector_fields = ["question_embedding", "answer_embedding"]
        else:
            # 为商品集合创建向量索引
            vector_fields = ["embedding"]
        
        for field in vector_fields:
            # has_index方法需要指定index_name作为关键字参数
            if collection.has_index(timeout=30, index_name=field):
                logger.info(f"索引 {field} 已存在")
            else:
                logger.info(f"为字段 {field} 创建索引...")
                collection.create_index(
                    field_name=field,
                    index_params=index_params,
                    timeout=30
                )
                logger.info(f"字段 {field} 索引创建完成")

    def load_embedding_data(self, data_dir: str = "embeddings_data") -> tuple:
        """加载embedding数据"""
        logger.info(f"从 {data_dir} 加载embedding数据...")
        
        # 加载向量数据
        product_embeddings = np.load(os.path.join(data_dir, "product_embeddings.npy"))
        question_embeddings = np.load(os.path.join(data_dir, "question_embeddings.npy"))
        answer_embeddings = np.load(os.path.join(data_dir, "answer_embeddings.npy"))
        
        # 加载处理后的数据
        with open(os.path.join(data_dir, "processed_products.json"), 'r', encoding='utf-8') as f:
            processed_products = json.load(f)
        
        with open(os.path.join(data_dir, "processed_qa_pairs.json"), 'r', encoding='utf-8') as f:
            processed_qa_pairs = json.load(f)
        
        logger.info(f"数据加载完成: {len(processed_products)}个商品, {len(processed_qa_pairs)}个问答对")
        
        return (
            product_embeddings, question_embeddings, answer_embeddings,
            processed_products, processed_qa_pairs
        )

    def prepare_product_data(self, processed_products: List[Dict], product_embeddings: np.ndarray) -> List[List]:
        """准备商品数据用于插入"""
        logger.info("准备商品数据...")
        
        # 按字段组织数据
        data = [
            [product["id"] for product in processed_products],  # id
            product_embeddings.tolist(),  # embedding
            [product["title"] for product in processed_products],  # title
            [product["description"] for product in processed_products],  # description
            [product["brand"] for product in processed_products],  # brand
            [product["category"] for product in processed_products],  # category
            [product["subcategory"] for product in processed_products],  # subcategory
            [float(product["price"]) for product in processed_products],  # price
            [product["condition"] for product in processed_products],  # condition
            [product["status"] for product in processed_products],  # status
            [product["location"] for product in processed_products],  # location
            [product["seller_id"] for product in processed_products],  # seller_id
            [int(product["views"]) for product in processed_products],  # views
            [int(product["likes"]) for product in processed_products],  # likes
            [product["created_at"] for product in processed_products],  # created_at
            [product["embedding_text"] for product in processed_products],  # embedding_text
        ]
        
        logger.info(f"商品数据准备完成: {len(data[0])} 条记录")
        return data

    def prepare_qa_data(self, processed_qa_pairs: List[Dict], 
                       question_embeddings: np.ndarray, 
                       answer_embeddings: np.ndarray) -> List[List]:
        """准备问答数据用于插入"""
        logger.info("准备问答数据...")
        
        # 按字段组织数据（注意：id字段是auto_id，不需要提供）
        data = [
            question_embeddings.tolist(),  # question_embedding
            answer_embeddings.tolist(),  # answer_embedding
            [qa["product_id"] for qa in processed_qa_pairs],  # product_id
            [qa["question"] for qa in processed_qa_pairs],  # question
            [qa["answer"] for qa in processed_qa_pairs],  # answer
            [qa["question_type"] for qa in processed_qa_pairs],  # question_type
            [qa["timestamp"] for qa in processed_qa_pairs],  # timestamp
            [qa["question_embedding_text"] for qa in processed_qa_pairs],  # question_embedding_text
            [qa["answer_embedding_text"] for qa in processed_qa_pairs],  # answer_embedding_text
        ]
        
        logger.info(f"问答数据准备完成: {len(data[0])} 条记录")
        return data

    def insert_data(self, collection: Collection, data: List[List], batch_size: int = 100):
        """批量插入数据"""
        total_records = len(data[0])
        logger.info(f"开始插入数据，总计 {total_records} 条记录")
        
        inserted_count = 0
        
        for i in range(0, total_records, batch_size):
            end_idx = min(i + batch_size, total_records)
            
            # 准备批次数据
            batch_data = []
            for field_data in data:
                batch_data.append(field_data[i:end_idx])
            
            # 插入数据
            insert_result = collection.insert(batch_data)
            inserted_count += len(insert_result.primary_keys)
            
            logger.info(f"已插入 {inserted_count}/{total_records} 条记录")
        
        # 刷新数据
        collection.flush()
        logger.info(f"数据插入完成，总计 {inserted_count} 条记录")
        
        return inserted_count

    def verify_data_integrity(self, collection: Collection, expected_count: int):
        """验证数据完整性"""
        logger.info("验证数据完整性...")
        
        # 获取集合统计信息
        collection.load()
        actual_count = collection.num_entities
        
        logger.info(f"期望记录数: {expected_count}")
        logger.info(f"实际记录数: {actual_count}")
        
        if actual_count == expected_count:
            logger.info("✅ 数据完整性验证通过")
            return True
        else:
            logger.error("❌ 数据完整性验证失败")
            return False

    def import_all_data(self, data_dir: str = "embeddings_data"):
        """导入所有数据的主函数"""
        try:
            # 1. 加载embedding数据
            (product_embeddings, question_embeddings, answer_embeddings,
             processed_products, processed_qa_pairs) = self.load_embedding_data(data_dir)
            
            # 2. 创建商品集合
            product_schema = self.create_product_collection_schema()
            product_collection = self.create_collection_if_not_exists(
                self.product_collection_name, product_schema
            )
            
            # 3. 创建问答集合
            qa_schema = self.create_qa_collection_schema()
            qa_collection = self.create_collection_if_not_exists(
                self.qa_collection_name, qa_schema
            )
            
            # 4. 创建索引
            self.create_indexes(product_collection, is_qa_collection=False)
            self.create_indexes(qa_collection, is_qa_collection=True)
            
            # 5. 准备并插入商品数据
            if product_collection.num_entities == 0:
                logger.info("插入商品数据...")
                product_data = self.prepare_product_data(processed_products, product_embeddings)
                self.insert_data(product_collection, product_data)
                self.verify_data_integrity(product_collection, len(processed_products))
            else:
                logger.info(f"商品集合已有 {product_collection.num_entities} 条记录，跳过插入")
            
            # 6. 准备并插入问答数据
            if qa_collection.num_entities == 0:
                logger.info("插入问答数据...")
                qa_data = self.prepare_qa_data(processed_qa_pairs, question_embeddings, answer_embeddings)
                self.insert_data(qa_collection, qa_data)
                self.verify_data_integrity(qa_collection, len(processed_qa_pairs))
            else:
                logger.info(f"问答集合已有 {qa_collection.num_entities} 条记录，跳过插入")
            
            # 7. 加载集合到内存
            product_collection.load()
            qa_collection.load()
            
            logger.info("🎉 所有数据导入完成！")
            
            # 输出统计信息
            self.print_statistics(product_collection, qa_collection)
            
        except Exception as e:
            logger.error(f"数据导入过程中出现错误: {e}")
            raise

    def print_statistics(self, product_collection: Collection, qa_collection: Collection):
        """打印统计信息"""
        print("\n" + "="*60)
        print("📊 Milvus数据库导入统计")
        print("="*60)
        print(f"🗄️  数据库名称: {self.db_name}")
        print(f"🔗 连接地址: {self.uri}")
        print(f"📦 商品集合: {self.product_collection_name}")
        print(f"   - 记录数: {product_collection.num_entities}")
        print(f"   - 向量维度: {self.vector_dim}")
        print(f"❓ 问答集合: {self.qa_collection_name}")
        print(f"   - 记录数: {qa_collection.num_entities}")
        print(f"   - 向量维度: {self.vector_dim}")
        print(f"⏰ 导入时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

    def test_search(self, query_text: str = "iPhone手机", top_k: int = 3):
        """测试搜索功能"""
        logger.info(f"测试搜索: '{query_text}'")
        
        try:
            # 这里需要重新生成query的embedding，但简化起见先跳过
            # 实际使用时需要加载embedding模型
            logger.info("搜索功能测试需要加载embedding模型，暂时跳过")
            
        except Exception as e:
            logger.error(f"搜索测试失败: {e}")

def main():
    """主函数"""
    # 从环境变量或直接配置连接信息
    importer = MilvusDataImporter(
        uri="http://43.140.216.157:18001",
        token="root:Haipro003838.",
        db_name="test_data",
        secure=False
    )
    
    try:
        # 导入所有数据
        importer.import_all_data("embeddings_data")
        
        # 简单测试（需要embedding模型才能完整测试）
        # importer.test_search("iPhone手机")
        
        logger.info("✅ 所有操作完成！")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()