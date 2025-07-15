"""Milvus向量数据库客户端类，提供向量存储和检索服务的访问接口。."""

import logging
import os
from typing import Dict, List
from urllib.parse import urlparse

from dotenv import load_dotenv
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    db,
    utility,
)

# 配置日志
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


class MilvusClient:
    """Milvus向量数据库客户端类，用于向量存储和检索服务的访问。.

    此类提供了一个统一的接口来访问Milvus向量数据库服务，包括连接管理、Collection管理和向量操作等功能。
    采用单例模式，全局共享同一个客户端实例。

    示例:
        ```python
        # 使用默认配置初始化客户端
        client = MilvusClient()

        # 创建Collection
        client.create_collection("documents", 1536)

        # 插入向量
        client.insert(
            "documents",
            vectors=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            ids=["doc1", "doc2"],
            metadata=[{"text": "文档1"}, {"text": "文档2"}],
        )

        # 搜索向量
        results = client.search(
            "documents", [[0.1, 0.2, ...]], limit=5, expr="text like '%文档%'"
        )
        ```
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """实现单例模式."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        uri: str | None = None,
        token: str | None = None,
        db_name: str | None = None,
        secure: bool | None = None,
        consistency_level: str | None = None,
    ):
        """初始化Milvus客户端。.

        参数:
            uri: Milvus服务器地址，默认从环境变量MILVUS_URI读取
            token: 访问令牌，默认从环境变量MILVUS_TOKEN读取
            db_name: 数据库名称，默认从环境变量MILVUS_DB_NAME读取
            secure: 是否使用HTTPS连接，默认从环境变量MILVUS_SECURE读取
            consistency_level: 一致性级别，默认从环境变量MILVUS_CONSISTENCY_LEVEL读取
        """
        if self._initialized:
            return

        # 从环境变量或参数中获取配置
        self.uri = uri or os.getenv("MILVUS_URI", "localhost:19530")
        self.token = token or os.getenv("MILVUS_TOKEN", "")
        self.db_name = db_name or os.getenv("MILVUS_DB_NAME", "default")

        # 解析secure参数
        if secure is None:
            secure_str = os.getenv("MILVUS_SECURE", "false").lower()
            self.secure = secure_str in ("true", "1", "yes")
        else:
            self.secure = secure

        # 解析一致性级别
        self.consistency_level = consistency_level or os.getenv(
            "MILVUS_CONSISTENCY_LEVEL", "Strong"
        )

        # 用户名和密码处理
        self.user = None
        self.password = None
        if self.token and ":" in self.token:
            self.user, self.password = self.token.split(":", 1)

        # 创建Milvus连接
        try:
            self._connect()
            logger.info(f"已成功连接到Milvus服务: {self.uri}")
            self._initialized = True
        except Exception as e:
            logger.error(f"连接Milvus服务失败: {str(e)}")
            raise

    def _connect(self):
        """建立与Milvus服务器的连接。."""
        try:
            # 处理URI：确保URI有正确的协议前缀
            uri = self.uri
            parsed_uri = None
            using_http_protocol = False
            connection_alias = "default"  # 默认连接别名

            # 如果URI以http://或https://开头，提取主机和端口
            if uri.startswith(("http://", "https://")):
                parsed_uri = urlparse(uri)
                host = parsed_uri.hostname
                # 修改此处：默认使用18001端口，而不是HTTP/HTTPS默认端口
                port = str(parsed_uri.port or 18001)  # 使用Milvus服务的实际端口
                using_http_protocol = True

                logger.info(f"检测到HTTP/HTTPS URI: {uri}")
                logger.info(f"提取的主机: {host}, 端口: {port}")

                # 如果使用自定义端口(18001)，不启用SSL，因为Milvus在该端口可能不支持SSL
                if port == "18001":
                    self.secure = False
                    logger.info(f"检测到端口18001，禁用SSL安全连接")
                # 对于标准HTTPS端口，保持启用SSL
                elif parsed_uri.scheme == "https" and (not parsed_uri.port or parsed_uri.port == 443):
                    self.secure = True
                    logger.info("启用SSL安全连接")

            # 识别协议类型
            supports_protocols = ["http://", "https://", "tcp://", "grpc://"]
            has_protocol = any(uri.startswith(p) for p in supports_protocols)

            # 检查是否仅包含主机:端口格式
            contains_port = ":" in uri and not has_protocol and not uri.startswith("//")

            # 确定连接方式:
            # 1. 如果是HTTP/HTTPS协议，尝试提取主机和端口直接连接
            # 2. 如果URI中包含协议前缀，直接使用URI
            # 3. 如果URI只有host:port格式，添加grpc://前缀
            if using_http_protocol and host and port:
                # 使用host和port连接，而不是原始URI
                logger.info(f"使用主机+端口连接: host={host}, port={port}")
                conn_params = {
                    "host": host,
                    "port": port,
                    "secure": self.secure,
                }
            elif has_protocol:
                # 对于已经有协议前缀的URI，直接使用
                logger.info(f"使用已有协议的URI连接: {uri}")
                conn_params = {
                    "uri": uri,
                    "secure": self.secure,
                }
            elif contains_port:
                # URI只包含主机:端口格式，添加grpc://前缀
                uri_with_protocol = f"grpc://{uri}"
                logger.info(f"URI没有协议前缀，添加grpc://: {uri_with_protocol}")
                conn_params = {
                    "uri": uri_with_protocol,
                    "secure": self.secure,
                }
            else:
                # 其他无法解析的URI格式，尝试直接使用
                logger.warning(f"无法解析的URI格式: {uri}，尝试直接使用")
                conn_params = {
                    "uri": uri,
                    "secure": self.secure,
                }

            # 添加用户名和密码
            if self.user and self.password:
                conn_params["user"] = self.user
                conn_params["password"] = self.password
                logger.info(f"使用用户名/密码验证: {self.user}")
            elif self.token and ":" in self.token:
                # 从token中提取用户名和密码
                user, password = self.token.split(":", 1)
                conn_params["user"] = user
                conn_params["password"] = password
                logger.info(f"从token提取用户名/密码: {user}")
            elif self.token:
                # 如果token不是用户名:密码格式，则视为完整token
                conn_params["token"] = self.token
                logger.info("使用token验证")

            # 添加数据库名称（如果可用）
            if self.db_name and self.db_name != "default":
                conn_params["db_name"] = self.db_name
                logger.info(f"指定数据库名称: {self.db_name}")

            # 添加连接别名
            conn_params["alias"] = connection_alias

            # 移除不必要的空参数
            conn_params = {k: v for k, v in conn_params.items() if v is not None}

            # 记录连接参数（隐藏密码）
            log_params = conn_params.copy()
            if "password" in log_params:
                log_params["password"] = "******"
            logger.info(f"连接Milvus服务，参数: {log_params}")

            # 检查是否已有连接，如果有则先断开
            try:
                if connections.has_connection(connection_alias):
                    logger.info(f"发现已有连接 {connection_alias}，先断开")
                    connections.disconnect(connection_alias)
            except Exception:
                pass

            # 尝试连接
            connections.connect(**conn_params)
            logger.info("Milvus连接成功!")

            # 如果指定了数据库，检查并使用
            if self.db_name != "default":
                try:
                    # 检查数据库是否存在
                    existing_dbs = db.list_database()
                    if self.db_name not in existing_dbs:
                        # 如果不存在，创建新数据库
                        db.create_database(self.db_name)
                        logger.info(f"已创建数据库: {self.db_name}")

                    # 使用指定的数据库
                    db.using_database(self.db_name)
                    logger.info(f"已切换到数据库: {self.db_name}")
                except Exception as e:
                    logger.warning(f"处理数据库 {self.db_name} 时出错: {str(e)}")

            # 存储连接别名
            self._connection_alias = connection_alias

            # 测试是否能列出集合 - 确认连接工作正常
            try:
                collections = utility.list_collections(using=connection_alias)
                logger.debug(f"连接测试成功，可用集合: {collections}")
            except Exception as e:
                logger.warning(f"列出集合测试失败，但连接可能依然有效: {str(e)}")

        except MilvusException as e:
            logger.error(f"连接Milvus服务时出错: {str(e)}")
            raise

    def close(self):
        """关闭与Milvus服务器的连接。."""
        try:
            # 使用存储的连接别名
            alias = getattr(self, "_connection_alias", "default")
            if connections.has_connection(alias):
                connections.disconnect(alias)
                logger.info(f"已断开与Milvus服务的连接 (别名: {alias})")
            else:
                logger.warning(f"没有找到活动连接 (别名: {alias})，无需断开")
        except Exception as e:
            logger.error(f"断开Milvus连接时出错: {str(e)}")

    # ===== Collection管理 =====

    def has_collection(self, collection_name: str) -> bool:
        """检查Collection是否存在。.

        参数:
            collection_name: Collection名称

        返回:
            如果Collection存在返回True，否则返回False
        """
        try:
            return utility.has_collection(collection_name)
        except MilvusException as e:
            logger.error(f"检查Collection {collection_name} 是否存在时出错: {str(e)}")
            return False

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        id_field: str = "id",
        vector_field: str = "vector",
        metric_type: str = "L2",
        index_type: str = "HNSW",
        index_params: Dict | None = None,
        description: str = "",
        auto_id: bool = False,
        consistency_level: str | None = None,
    ) -> bool:
        """创建一个新的Collection。.

        参数:
            collection_name: Collection名称
            dimension: 向量维度
            id_field: ID字段名称，默认为"id"
            vector_field: 向量字段名称，默认为"vector"
            metric_type: 距离度量类型，默认为"L2"，可选值包括"L2"、"IP"、"COSINE"等
            index_type: 索引类型，默认为"HNSW"
            index_params: 索引参数，默认为None，会根据索引类型设置合理默认值
            description: Collection描述，默认为空字符串
            auto_id: 是否自动生成ID，默认为False
            consistency_level: 一致性级别，默认使用客户端配置的值

        返回:
            如果成功创建返回True，否则返回False
        """
        try:
            # 先检查Collection是否已存在
            if self.has_collection(collection_name):
                logger.info(f"Collection {collection_name} 已存在")
                return True

            # 定义主键字段
            fields = [
                FieldSchema(
                    name=id_field,
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=auto_id,
                    max_length=100,
                ),
                # 元数据JSON字段，用于存储额外信息
                FieldSchema(
                    name="metadata",
                    dtype=DataType.JSON,
                    description="存储元数据的JSON字段",
                ),
                # 向量字段
                FieldSchema(
                    name=vector_field,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dimension,
                    description=f"{dimension}维向量字段",
                ),
            ]

            # 创建schema
            schema = CollectionSchema(fields=fields, description=description)

            # 创建Collection
            consistency = consistency_level or self.consistency_level
            collection = Collection(
                name=collection_name, schema=schema, consistency_level=consistency
            )
            logger.info(f"已成功创建Collection: {collection_name}")

            # 设置索引
            if index_type:
                # 设置默认索引参数
                default_params = {
                    "HNSW": {"M": 16, "efConstruction": 200},
                    "IVF_FLAT": {"nlist": 128},
                    "IVF_SQ8": {"nlist": 128},
                    "IVF_PQ": {"nlist": 128, "m": 8},
                    "FLAT": {},
                    "ANNOY": {"n_trees": 8},
                }

                # 合并用户提供的索引参数
                params = index_params or {}
                if index_type in default_params and not params:
                    params = default_params[index_type]

                # 创建索引
                collection.create_index(
                    field_name=vector_field,
                    index_params={
                        "index_type": index_type,
                        "metric_type": metric_type,
                        "params": params,
                    },
                )
                logger.info(f"已为Collection {collection_name} 创建索引: {index_type}")

            return True

        except MilvusException as e:
            logger.error(f"创建Collection {collection_name} 时出错: {str(e)}")
            return False

    def list_collections(self) -> List[str]:
        """列出所有可用的Collection。.

        返回:
            Collection名称列表
        """
        try:
            # 使用存储的连接别名
            alias = getattr(self, "_connection_alias", "default")
            # 确保连接存在
            if not connections.has_connection(alias):
                logger.warning(f"连接 {alias} 不存在，尝试重新连接")
                # 保留原来的连接参数进行重连
                self._connect()

            return utility.list_collections(using=alias)
        except MilvusException as e:
            logger.error(f"列出Collections时出错: {str(e)}")
            return []

    def drop_collection(self, collection_name: str) -> bool:
        """删除一个Collection。.

        参数:
            collection_name: Collection名称

        返回:
            如果成功删除返回True，否则返回False
        """
        try:
            # 先检查Collection是否存在
            if not self.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} 不存在")
                return False

            # 删除Collection
            utility.drop_collection(collection_name)
            logger.info(f"已成功删除Collection: {collection_name}")
            return True
        except MilvusException as e:
            logger.error(f"删除Collection {collection_name} 时出错: {str(e)}")
            return False

    def load_collection(self, collection_name: str) -> bool:
        """将Collection加载到内存中，以便进行搜索。.

        参数:
            collection_name: Collection名称

        返回:
            如果成功加载返回True，否则返回False
        """
        try:
            # 先检查Collection是否存在
            if not self.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} 不存在")
                return False

            # 加载Collection
            collection = Collection(collection_name)
            collection.load()
            logger.info(f"已成功加载Collection: {collection_name}")
            return True
        except MilvusException as e:
            logger.error(f"加载Collection {collection_name} 时出错: {str(e)}")
            return False

    def get_collection_info(self, collection_name: str) -> Dict | None:
        """获取Collection的详细信息。.

        参数:
            collection_name: Collection名称

        返回:
            包含Collection信息的字典，如果出错则返回None
        """
        try:
            # 先检查Collection是否存在
            if not self.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} 不存在")
                return None

            # 获取Collection信息
            collection = Collection(collection_name)

            # 获取基本信息
            info = {
                "name": collection.name,
                "description": collection.description,
                "schema": collection.schema,
                "num_entities": collection.num_entities,
                "indexes": [],
            }

            # 获取索引信息
            try:
                indexes = collection.index().params
                info["indexes"] = indexes
            except IndexError:
                pass

            return info
        except MilvusException as e:
            logger.error(f"获取Collection {collection_name} 信息时出错: {str(e)}")
            return None

    def count_entities(self, collection_name: str) -> int:
        """获取Collection中实体的数量。.

        参数:
            collection_name: Collection名称

        返回:
            实体数量，如果出错则返回0
        """
        try:
            # 先检查Collection是否存在
            if not self.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} 不存在")
                return 0

            # 获取实体数量
            collection = Collection(collection_name)
            return collection.num_entities
        except MilvusException as e:
            logger.error(f"获取Collection {collection_name} 实体数量时出错: {str(e)}")
            return 0

    # ===== 向量操作 =====

    def insert(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: List[str] | None = None,
        metadata: List[Dict] | None = None,
        vector_field: str = "vector",
        id_field: str = "id",
    ) -> List[str] | None:
        """向Collection中插入向量数据。.

        参数:
            collection_name: Collection名称
            vectors: 向量数据列表
            ids: ID列表，长度必须与vectors相同
            metadata: 元数据列表，长度必须与vectors相同
            vector_field: 向量字段名称，默认为"vector"
            id_field: ID字段名称，默认为"id"

        返回:
            插入的ID列表，如果出错则返回None
        """
        try:
            # 先检查Collection是否存在
            if not self.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} 不存在")
                return None

            # 创建Collection对象
            collection = Collection(collection_name)

            # 检查ID列表
            if ids is None:
                # 如果没有提供ID，则生成UUID
                import uuid

                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

            # 检查metadata列表
            if metadata is None:
                # 如果没有提供metadata，则创建空字典列表
                metadata = [{} for _ in range(len(vectors))]

            # 检查长度是否一致
            if len(vectors) != len(ids) or len(vectors) != len(metadata):
                error_msg = "vectors、ids和metadata的长度必须相同"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # 准备插入数据
            insert_data = [
                ids,  # id_field
                metadata,  # metadata
                vectors,  # vector_field
            ]

            # 执行插入
            result = collection.insert(insert_data)

            # 获取插入的ID
            inserted_ids = result.primary_keys
            logger.info(
                f"已成功向Collection {collection_name} 插入 {len(inserted_ids)} 条数据"
            )

            return inserted_ids
        except MilvusException as e:
            logger.error(f"向Collection {collection_name} 插入数据时出错: {str(e)}")
            return None
        except Exception as e:
            logger.error(
                f"向Collection {collection_name} 插入数据时发生未知错误: {str(e)}"
            )
            return None

    def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        limit: int = 10,
        expr: str | None = None,
        output_fields: List[str] | None = None,
        vector_field: str = "vector",
        metric_type: str | None = None,
        params: Dict | None = None,
    ) -> List[List[Dict]]:
        """在Collection中搜索相似向量。.

        参数:
            collection_name: Collection名称
            query_vectors: 查询向量列表
            limit: 返回的最大结果数，默认为10
            expr: 过滤表达式，默认为None
            output_fields: 返回的字段列表，默认为None (返回所有字段)
            vector_field: 向量字段名称，默认为"vector"
            metric_type: 距离度量类型，默认为None (使用Collection设置)
            params: 搜索参数，默认为None

        返回:
            搜索结果列表，每个查询向量对应一个结果列表
        """
        try:
            # 先检查Collection是否存在
            if not self.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} 不存在")
                return []

            # 创建Collection对象
            collection = Collection(collection_name)

            # 确保Collection已加载
            if not collection.is_loaded:
                collection.load()

            # 准备输出字段，默认包含id和metadata
            if output_fields is None:
                output_fields = ["id", "metadata"]
            elif "id" not in output_fields:
                output_fields.append("id")

            # 准备搜索参数
            search_params = {}
            if metric_type:
                search_params["metric_type"] = metric_type
            if params:
                search_params["params"] = params

            # 执行搜索
            results = collection.search(
                data=query_vectors,
                anns_field=vector_field,
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=output_fields,
            )

            # 处理结果
            processed_results = []
            for hits in results:
                hits_list = []
                for hit in hits:
                    # 构建结果字典
                    hit_dict = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "score": 1.0 / (1.0 + hit.distance),  # 简单的分数转换
                    }

                    # 添加输出字段
                    for field in output_fields:
                        if field != "id" and hasattr(hit.entity, field):
                            hit_dict[field] = getattr(hit.entity, field)

                    hits_list.append(hit_dict)
                processed_results.append(hits_list)

            logger.info(
                f"已在Collection {collection_name} 中搜索到 {sum(len(r) for r in processed_results)} 条结果"
            )
            return processed_results

        except MilvusException as e:
            logger.error(f"在Collection {collection_name} 中搜索时出错: {str(e)}")
            return []
        except Exception as e:
            logger.error(
                f"在Collection {collection_name} 中搜索时发生未知错误: {str(e)}"
            )
            return []

    def query(
        self,
        collection_name: str,
        expr: str,
        output_fields: List[str] | None = None,
        limit: int = 100,
    ) -> List[Dict]:
        """在Collection中基于表达式查询数据。.

        参数:
            collection_name: Collection名称
            expr: 查询表达式
            output_fields: 返回的字段列表，默认为None (返回所有字段)
            limit: 返回的最大结果数，默认为100

        返回:
            查询结果列表
        """
        try:
            # 先检查Collection是否存在
            if not self.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} 不存在")
                return []

            # 创建Collection对象
            collection = Collection(collection_name)

            # 确保Collection已加载
            if not collection.is_loaded:
                collection.load()

            # 准备输出字段
            if output_fields is None:
                output_fields = ["id", "metadata"]

            # 执行查询
            results = collection.query(
                expr=expr, output_fields=output_fields, limit=limit
            )

            logger.info(
                f"已在Collection {collection_name} 中查询到 {len(results)} 条结果"
            )
            return results

        except MilvusException as e:
            logger.error(f"在Collection {collection_name} 中查询时出错: {str(e)}")
            return []
        except Exception as e:
            logger.error(
                f"在Collection {collection_name} 中查询时发生未知错误: {str(e)}"
            )
            return []

    def delete(self, collection_name: str, expr: str) -> bool:
        """从Collection中删除数据。.

        参数:
            collection_name: Collection名称
            expr: 删除表达式，例如"id in ['id1', 'id2']"

        返回:
            如果成功删除返回True，否则返回False
        """
        try:
            # 先检查Collection是否存在
            if not self.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} 不存在")
                return False

            # 创建Collection对象
            collection = Collection(collection_name)

            # 执行删除
            collection.delete(expr)
            logger.info(
                f"已从Collection {collection_name} 中删除满足条件 '{expr}' 的数据"
            )

            return True

        except MilvusException as e:
            logger.error(f"从Collection {collection_name} 中删除数据时出错: {str(e)}")
            return False
        except Exception as e:
            logger.error(
                f"从Collection {collection_name} 中删除数据时发生未知错误: {str(e)}"
            )
            return False
