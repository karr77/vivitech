"""Milvus客户端的真实连接测试。."""

import logging
import os
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径，解决导入问题
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pymilvus import connections, utility

from client import MilvusClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("milvus_test")


class TestMilvusConnection(unittest.TestCase):
    """测试Milvus的各种连接方式。."""

    def setUp(self):
        """设置测试环境。."""
        # 连接参数 - 使用成功验证过的配置
        self.milvus_uri = os.getenv("MILVUS_URI", "http://43.140.216.157:18001")
        self.milvus_token = os.getenv("MILVUS_TOKEN", "root:Haipro003838.")
        self.milvus_db = os.getenv("MILVUS_DB_NAME", "deer_conversation_history")
        self.milvus_secure = os.getenv("MILVUS_SECURE", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        # 打印环境变量配置
        logger.info("测试连接参数:")
        logger.info(f"MILVUS_URI = {self.milvus_uri}")
        logger.info(f"MILVUS_DB_NAME = {self.milvus_db}")
        logger.info(f"MILVUS_TOKEN = {'已设置' if self.milvus_token else '未设置'}")
        logger.info(f"MILVUS_SECURE = {self.milvus_secure}")

        # 提取host和port (用于host-port连接方式)
        from urllib.parse import urlparse

        parsed_uri = urlparse(self.milvus_uri)
        self.host = parsed_uri.hostname or "43.140.216.157"
        self.port = str(parsed_uri.port or 18001)

        # 提取用户名和密码
        self.user = None
        self.password = None
        if self.milvus_token and ":" in self.milvus_token:
            self.user, self.password = self.milvus_token.split(":", 1)

        # 创建唯一测试集合名称
        self.timestamp = int(time.time())
        self.test_collection = f"test_connection_{self.timestamp}"

        # 每个测试类使用唯一的连接别名前缀
        self.conn_prefix = "conn_basic"

    def tearDown(self):
        """清理连接。"""
        # 断开所有可能的连接
        for alias in [f"{self.conn_prefix}_host_port", f"{self.conn_prefix}_client"]:
            try:
                if connections.has_connection(alias):
                    connections.disconnect(alias)
                    logger.info(f"已断开连接: {alias}")
            except Exception as e:
                logger.warning(f"断开连接 {alias} 时出错: {e}")

    def test_host_port_connection(self):
        """测试使用host和port方式连接。."""
        conn_alias = f"{self.conn_prefix}_host_port"
        logger.info("=" * 80)
        logger.info(f"开始测试【远程穿透连接】(使用host和port方式，别名: {conn_alias})")

        try:
            # 先确保没有旧连接
            if connections.has_connection(conn_alias):
                connections.disconnect(conn_alias)
                logger.info(f"已断开旧连接: {conn_alias}")

            # 使用host和port方式连接
            conn_params = {
                "alias": conn_alias,
                "host": self.host,
                "port": self.port,
                "user": self.user,
                "password": self.password,
                "db_name": self.milvus_db,
            }

            logger.info(f"连接参数: {conn_params}")
            connections.connect(**conn_params)

            # 验证连接
            version = utility.get_server_version(conn_alias)
            collections = utility.list_collections(using=conn_alias)

            logger.info("✅ 远程穿透连接成功!")
            logger.info(f"服务器版本: {version}")
            logger.info(f"可用集合: {collections}")

            # 不要在这里断开连接，在tearDown中统一处理
            self.assertTrue(True)  # 如果执行到这里，表示测试成功

        except Exception as e:
            logger.error(f"❌ 远程穿透连接失败: {e}")
            self.fail(f"远程穿透连接失败: {e}")

    def test_milvus_client_connection(self):
        """测试使用MilvusClient类连接。."""
        conn_alias = f"{self.conn_prefix}_client"
        logger.info("=" * 80)
        logger.info(f"开始测试【MilvusClient类连接】(别名: {conn_alias})")

        try:
            # 使用MilvusClient类连接，指定连接别名
            client = MilvusClient(
                uri=self.milvus_uri,
                token=self.milvus_token,
                db_name=self.milvus_db,
                secure=self.milvus_secure,
            )
            client._connection_alias = conn_alias  # 强制使用我们的连接别名

            # 验证连接
            collections = client.list_collections()
            logger.info("✅ MilvusClient类连接成功!")
            logger.info(f"可用集合: {collections}")

            # 测试创建集合
            test_collection = f"test_client_{self.timestamp}"
            logger.info(f"创建测试集合: {test_collection}")
            client.create_collection(
                collection_name=test_collection,
                dimension=1536,  # OpenAI embedding-3-small 维度
                description=f"测试集合，创建于 {datetime.now().isoformat()}",
            )

            # 验证集合创建成功
            self.assertTrue(client.has_collection(test_collection))
            logger.info(f"测试集合创建成功: {test_collection}")

            # 获取集合信息
            collection_info = client.get_collection_info(test_collection)
            logger.info(f"集合信息: {collection_info}")

            # 关闭连接
            client.close()
            self.assertTrue(True)  # 如果执行到这里，表示测试成功

        except Exception as e:
            logger.error(f"❌ MilvusClient类连接失败: {e}")
            self.fail(f"MilvusClient类连接失败: {e}")


class SimpleMilvusClientTests(unittest.TestCase):
    """简化的MilvusClient测试类，专注于连接和基本操作。"""

    def setUp(self):
        """设置测试环境。"""
        # 使用成功验证过的连接参数
        self.milvus_uri = os.getenv("MILVUS_URI", "http://43.140.216.157:18001")
        self.milvus_token = os.getenv("MILVUS_TOKEN", "root:Haipro003838.")
        self.milvus_db = os.getenv("MILVUS_DB_NAME", "deer_conversation_history")
        self.milvus_secure = os.getenv("MILVUS_SECURE", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        # 提取用户名和密码
        self.user = None
        self.password = None
        if self.milvus_token and ":" in self.milvus_token:
            self.user, self.password = self.milvus_token.split(":", 1)

        # 创建唯一测试集合名称和连接别名
        self.timestamp = int(time.time())
        self.test_collection = f"test_simple_{self.timestamp}"
        self.conn_alias = f"simple_test_{self.timestamp}"

        logger.info(
            f"初始化SimpleMilvusClientTests, 集合名: {self.test_collection}, 连接别名: {self.conn_alias}"
        )

    def tearDown(self):
        """清理环境。"""
        try:
            if connections.has_connection(self.conn_alias):
                connections.disconnect(self.conn_alias)
                logger.info(f"已断开连接: {self.conn_alias}")
        except Exception as e:
            logger.warning(f"断开连接 {self.conn_alias} 时出错: {e}")

    def test_basic_operations(self):
        """测试基本操作：连接、创建集合、插入数据、搜索。"""
        logger.info("=" * 80)
        logger.info(f"开始基本操作测试，使用连接别名: {self.conn_alias}")

        try:
            # 1. 直接使用pymilvus连接
            from urllib.parse import urlparse

            parsed_uri = urlparse(self.milvus_uri)
            host = parsed_uri.hostname or "43.140.216.157"
            port = str(parsed_uri.port or 18001)

            conn_params = {
                "alias": self.conn_alias,
                "host": host,
                "port": port,
                "user": self.user,
                "password": self.password,
                "db_name": self.milvus_db,
            }

            logger.info(f"连接参数: {conn_params}")
            connections.connect(**conn_params)

            # 2. 验证连接
            version = utility.get_server_version(self.conn_alias)
            collections = utility.list_collections(using=self.conn_alias)
            logger.info(f"连接成功! 服务器版本: {version}")
            logger.info(f"可用集合: {collections}")

            # 3. 创建测试集合
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

            # 检查集合是否已存在
            if utility.has_collection(self.test_collection, using=self.conn_alias):
                utility.drop_collection(self.test_collection, using=self.conn_alias)
                logger.info(f"已删除已存在的集合: {self.test_collection}")

            # 定义集合字段
            fields = [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100
                ),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(
                    name="vector", dtype=DataType.FLOAT_VECTOR, dim=16
                ),  # 使用小维度便于测试
            ]

            schema = CollectionSchema(fields=fields, description="简单测试集合")
            collection = Collection(
                name=self.test_collection, schema=schema, using=self.conn_alias
            )
            logger.info(f"集合创建成功: {self.test_collection}")

            # 4. 创建索引
            index_params = {
                "metric_type": "L2",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64},
            }
            collection.create_index(field_name="vector", index_params=index_params)
            logger.info("索引创建成功")

            # 5. 插入数据
            test_vectors = [[float(i) * 0.1 for i in range(16)] for _ in range(2)]
            test_ids = ["test_id_1", "test_id_2"]
            test_metadata = [{"text": "测试1"}, {"text": "测试2"}]

            insert_data = [test_ids, test_metadata, test_vectors]
            collection.insert(insert_data)
            logger.info("数据插入成功，正在加载集合...")

            # 6. 加载集合
            collection.load()
            logger.info(f"集合已加载，实体数量: {collection.num_entities}")

            # 7. 搜索
            search_params = {"metric_type": "L2", "params": {"ef": 10}}
            results = collection.search(
                data=[test_vectors[0]],
                anns_field="vector",
                param=search_params,
                limit=5,
                output_fields=["metadata"],
            )

            logger.info(f"搜索完成，结果数量: {len(results)}")
            if results and results[0]:
                logger.info(
                    f"搜索结果第一个: ID={results[0][0].id}, 距离={results[0][0].distance}"
                )

            self.assertTrue(len(results) > 0)
            self.assertTrue(len(results[0]) > 0)

        except Exception as e:
            logger.error(f"基本操作测试失败: {e}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            self.fail(f"基本操作测试失败: {e}")


class SimpleVectorStoreTests(unittest.TestCase):
    """简化的MilvusVectorStore测试类，专注于基本功能。"""

    def setUp(self):
        """设置测试环境。"""
        self.milvus_uri = os.getenv("MILVUS_URI", "http://43.140.216.157:18001")
        self.milvus_token = os.getenv("MILVUS_TOKEN", "root:Haipro003838.")
        self.milvus_db = os.getenv("MILVUS_DB_NAME", "deer_conversation_history")
        self.milvus_secure = os.getenv("MILVUS_SECURE", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        openai_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_base = os.getenv("OPENAI_BASE_URL", "https://jp.rcouyi.com/v1")

        # 创建唯一测试集合名称和连接别名
        self.timestamp = int(time.time())
        self.test_collection = f"vs_test_{self.timestamp}"
        self.conn_alias = f"vs_test_{self.timestamp}"

        # 检查OpenAI API密钥是否设置
        if not openai_key:
            self.skipTest("OPENAI_API_KEY未设置，跳过向量存储测试")
            return

        logger.info(
            f"初始化SimpleVectorStoreTests, 集合名: {self.test_collection}, 连接别名: {self.conn_alias}"
        )

        try:
            # 创建嵌入模型
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                dimensions=1536,
                api_key=openai_key,
                base_url=self.openai_base,
            )
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {e}")
            self.skipTest(f"初始化嵌入模型失败: {e}")
            return

        # 创建测试文档
        self.test_docs = [
            Document(
                page_content="向量数据库是存储和检索向量数据的专用数据库系统",
                metadata={"source": "tech", "type": "definition"},
            ),
            Document(
                page_content="Milvus是一个开源的向量数据库，专为嵌入式向量相似度搜索设计",
                metadata={"source": "tech", "type": "product"},
            ),
        ]

    def tearDown(self):
        """清理环境。"""
        try:
            if connections.has_connection(self.conn_alias):
                connections.disconnect(self.conn_alias)
                logger.info(f"已断开连接: {self.conn_alias}")
        except Exception as e:
            logger.warning(f"断开连接 {self.conn_alias} 时出错: {e}")

    def test_direct_pymilvus_vector_store(self):
        """使用直接的pymilvus连接和MilvusVectorStore测试。"""
        logger.info("=" * 80)
        logger.info(f"开始直接pymilvus向量存储测试，使用连接别名: {self.conn_alias}")

        try:
            # 1. 首先建立pymilvus连接
            from urllib.parse import urlparse

            parsed_uri = urlparse(self.milvus_uri)
            host = parsed_uri.hostname or "43.140.216.157"
            port = str(parsed_uri.port or 18001)

            user = None
            password = None
            if self.milvus_token and ":" in self.milvus_token:
                user, password = self.milvus_token.split(":", 1)

            conn_params = {
                "alias": self.conn_alias,
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "db_name": self.milvus_db,
            }

            logger.info(f"连接参数: {conn_params}")
            connections.connect(**conn_params)

            # 2. 验证连接
            version = utility.get_server_version(self.conn_alias)
            collections = utility.list_collections(using=self.conn_alias)
            logger.info(f"连接成功! 服务器版本: {version}")
            logger.info(f"可用集合: {collections}")

            # 3. 使用MilvusVectorStore，手动传入连接别名参数
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

            # 如果集合已存在，先删除
            if utility.has_collection(self.test_collection, using=self.conn_alias):
                utility.drop_collection(self.test_collection, using=self.conn_alias)
                logger.info(f"已删除已存在的集合: {self.test_collection}")

            # 4. 创建向量存储并添加文档
            # 手动创建集合
            text_field = "text"
            vector_field = "vector"
            meta_field = "metadata"

            fields = [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100
                ),
                FieldSchema(name=text_field, dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name=meta_field, dtype=DataType.JSON),
                FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, dim=1536),
            ]

            schema = CollectionSchema(fields=fields, description="向量存储测试集合")
            collection = Collection(
                name=self.test_collection, schema=schema, using=self.conn_alias
            )
            logger.info(f"集合创建成功: {self.test_collection}")

            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64},
            }
            collection.create_index(field_name=vector_field, index_params=index_params)
            logger.info("索引创建成功")

            # 5. 嵌入文档并插入
            texts = [doc.page_content for doc in self.test_docs]
            metadatas = [doc.metadata for doc in self.test_docs]

            # 生成嵌入向量
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(
                f"生成了 {len(embeddings)} 个嵌入向量，维度: {len(embeddings[0])}"
            )

            # 生成ID
            import uuid

            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

            # 准备插入数据
            insert_data = [ids, texts, metadatas, embeddings]

            collection.insert(insert_data)
            logger.info(f"数据插入成功，集合实体数量: {collection.num_entities}")

            # 6. 加载集合并搜索
            collection.load()
            logger.info("集合已加载")

            # 搜索
            query = "向量数据库的作用是什么"
            query_vector = self.embeddings.embed_query(query)

            search_params = {"metric_type": "COSINE", "params": {"ef": 10}}
            results = collection.search(
                data=[query_vector],
                anns_field=vector_field,
                param=search_params,
                limit=2,
                output_fields=[text_field, meta_field],
            )

            logger.info(f"搜索完成，结果数量: {len(results)}")
            if results and results[0]:
                logger.info(
                    f"搜索结果第一个: ID={results[0][0].id}, 距离={results[0][0].distance}"
                )
                logger.info(f"文本: {results[0][0].entity.get(text_field)}")
                logger.info(f"元数据: {results[0][0].entity.get(meta_field)}")

            self.assertTrue(len(results) > 0)
            self.assertTrue(len(results[0]) > 0)

        except Exception as e:
            logger.error(f"向量存储测试失败: {e}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            self.fail(f"向量存储测试失败: {e}")


if __name__ == "__main__":
    unittest.main()
