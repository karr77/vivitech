"""基于Milvus的商品知识库实现."""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .client import MilvusClient
from sentence_transformers import SentenceTransformer

# 配置日志
logger = logging.getLogger(__name__)

class MilvusKnowledgeBase:
    """基于Milvus的商品知识库，用于存储和检索闲鱼商品信息."""
    
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-large-zh-v1.5",
                 milvus_uri: Optional[str] = None,
                 milvus_token: Optional[str] = None,
                 milvus_db: Optional[str] = None):
        """初始化商品知识库.
        
        Args:
            embedding_model: 用于文本嵌入的模型名称
            milvus_uri: Milvus服务URI，默认从环境变量读取
            milvus_token: Milvus访问令牌，默认从环境变量读取
            milvus_db: Milvus数据库名称，默认从环境变量读取
        """
        # 初始化Milvus客户端
        self.client = MilvusClient(
            uri=milvus_uri, 
            token=milvus_token, 
            db_name=milvus_db
        )
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_dimension = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"当前嵌入模型维度: {self.vector_dimension}")
        
        # 集合名称定义
        # 新的分离集合
        self.items_collection = "items_collection"     # 商品基本信息
        self.images_collection = "images_collection"   # 商品图片信息  
        
        # 确保集合存在
        self._ensure_collections_exist()
        
    def _ensure_collections_exist(self):
        """确保所需的集合已存在，不存在则创建."""
        collections_to_check = [
            (self.items_collection, "闲鱼商品基本信息集合"),
            (self.images_collection, "闲鱼商品图片信息集合")
        ]
        
        for collection_name, description in collections_to_check:
            # 检查集合是否存在
            if self.client.has_collection(collection_name):
                # 检查维度是否匹配
                try:
                    collection_info = self.client.get_collection_info(collection_name)
                    current_dim = collection_info.get("dimension", 0)
                    
                    if current_dim != self.vector_dimension and current_dim != 0:
                        logger.warning(f"集合 {collection_name} 的向量维度 ({current_dim}) 与当前模型维度 ({self.vector_dimension}) 不匹配，正在删除并重建...")
                        self.client.drop_collection(collection_name)
                        # 集合将在下面重新创建
                    else:
                        logger.info(f"集合 {collection_name} 已存在且维度匹配，跳过创建")
                        continue
                except Exception as e:
                    logger.error(f"检查集合 {collection_name} 维度时出错: {e}")
            
            # 创建集合（如果不存在或已被删除）
            try:
                self.client.create_collection(
                    collection_name=collection_name,
                    dimension=self.vector_dimension,
                    description=description,
                    metric_type="COSINE",
                    index_type="HNSW",
                )
                logger.info(f"已创建集合: {collection_name}，向量维度: {self.vector_dimension}")
            except Exception as e:
                logger.error(f"创建集合 {collection_name} 时出错: {e}")
    
    def _embed_text(self, text: str) -> List[float]:
        """将文本转换为向量嵌入.
        
        Args:
            text: 需要嵌入的文本
            
        Returns:
            文本的向量嵌入
        """
        # 使用SentenceTransformer生成嵌入
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def import_multi_collection(self, items_data: List[Dict], force_update: bool = False) -> Dict[str, int]:
        """将商品数据导入到多个专用集合.
        
        Args:
            items_data: 商品数据列表，每项包含所有商品信息
            force_update: 是否强制更新已存在的数据
            
        Returns:
            Dict[str, int]: 各集合成功导入的数据条数
        """
        if not items_data:
            logger.warning("没有商品数据需要导入")
            return {"items": 0, "images": 0}
            
        results = {
            "items": 0,
            "images": 0
        }
        
        # 处理商品基本信息集合数据
        item_ids = []
        item_vectors = []
        item_metadata_list = []
        
        # 处理图片集合数据
        image_ids = []
        image_vectors = []
        image_metadata_list = []
        
        for item_data in items_data:
            item_id = item_data.get("itemId")
            if not item_id:
                logger.warning("跳过没有itemId的商品数据")
                continue
            
            title = item_data.get("title", "")
            content = item_data.get("content", "")
            price = float(item_data.get("price", 0.0))
            status = item_data.get("status", "unknown")
            image_url = item_data.get("imageUrl", "")
            category = item_data.get("category", "basic_info")
            timestamp = datetime.now().isoformat()
            
            # 1. 处理商品基本信息
            if category == "basic_info":
                # 生成嵌入文本并创建向量
                embed_text = f"{title} {content}"
                item_vector = self._embed_text(embed_text)
                
                # 构建元数据
                item_metadata = {
                    "itemId": item_id,
                    "title": title,
                    "content": content,
                    "price": price,
                    "status": status,
                    "lastUpdated": timestamp
                }
                
                # 如果强制更新，先删除已存在的项
                if force_update:
                    self.client.delete(
                        collection_name=self.items_collection,
                        expr=f'metadata["itemId"] == "{item_id}"'
                    )
                
                # 添加到批量处理列表
                item_ids.append(f"{item_id}_info")
                item_vectors.append(item_vector)
                item_metadata_list.append(item_metadata)
                
                # 2. 处理商品图片信息(如果有)
                if image_url:
                    # 图片描述文本
                    image_text = f"{title} 商品图片"
                    image_vector = self._embed_text(image_text)
                    
                    # 图片元数据
                    image_metadata = {
                        "itemId": item_id,
                        "title": title,
                        "imageUrl": image_url,
                        "status": status,
                        "lastUpdated": timestamp
                    }
                    
                    # 如果强制更新，先删除已存在的图片
                    if force_update:
                        self.client.delete(
                            collection_name=self.images_collection,
                            expr=f'metadata["itemId"] == "{item_id}"'
                        )
                    
                    # 添加到批量处理列表
                    image_ids.append(f"{item_id}_image")
                    image_vectors.append(image_vector)
                    image_metadata_list.append(image_metadata)
        
        # 批量插入商品基本信息
        if item_ids:
            try:
                result = self.client.insert(
                    collection_name=self.items_collection,
                    vectors=item_vectors,
                    ids=item_ids,
                    metadata=item_metadata_list
                )
                results["items"] = len(result) if result else 0
                logger.info(f"成功导入 {results['items']} 条商品基本信息")
            except Exception as e:
                logger.error(f"导入商品基本信息出错: {e}")
        
        # 批量插入图片信息  
        if image_ids:
            try:
                result = self.client.insert(
                    collection_name=self.images_collection,
                    vectors=image_vectors,
                    ids=image_ids,
                    metadata=image_metadata_list
                )
                results["images"] = len(result) if result else 0
                logger.info(f"成功导入 {results['images']} 条商品图片信息")
            except Exception as e:
                logger.error(f"导入商品图片信息出错: {e}")
        
        return results
    
    def remove_items(self, item_ids: List[str]) -> Dict[str, int]:
        """从所有集合中批量删除指定商品ID的数据

        Args:
            item_ids: 商品ID列表

        Returns:
            Dict[str, int]: 各集合中删除的记录数量
        """
        if not item_ids:
            return {"items": 0, "images": 0}
            
        results = {
            "items": 0,    # 商品基本信息集合
            "images": 0    # 图片集合
        }
            
        try:
            # 构建删除表达式
            expr_parts = [f'metadata["itemId"] == "{item_id}"' for item_id in item_ids]
            expr = " || ".join(expr_parts)
                
            # 从商品基本信息集合中删除
            try:
                # 先统计删除前的数量
                before_count = self.client.count_entities(self.items_collection)
                delete_success = self.client.delete(
                    collection_name=self.items_collection,
                    expr=expr
                )
                if delete_success:
                    # 删除后重新统计，计算差值
                    after_count = self.client.count_entities(self.items_collection)
                    results["items"] = before_count - after_count if before_count >= after_count else 0
                    logger.info(f"从商品基本信息集合中删除了 {results['items']} 条记录")
                else:
                    logger.warning(f"从商品基本信息集合删除数据失败")
            except Exception as e:
                logger.error(f"从商品基本信息集合删除数据出错: {e}")
                
            # 从图片集合中删除
            try:
                # 先统计删除前的数量
                before_count = self.client.count_entities(self.images_collection)
                delete_success = self.client.delete(
                    collection_name=self.images_collection,
                    expr=expr
                )
                if delete_success:
                    # 删除后重新统计，计算差值
                    after_count = self.client.count_entities(self.images_collection)
                    results["images"] = before_count - after_count if before_count >= after_count else 0
                    logger.info(f"从图片集合中删除了 {results['images']} 条记录")
                else:
                    logger.warning(f"从图片集合删除数据失败")
            except Exception as e:
                logger.error(f"从图片集合删除数据出错: {e}")
                
            return results
                
        except Exception as e:
            logger.error(f"批量删除商品失败: {e}")
            return results
    
    def search_items(self, query: str, limit: int = 5, collection_name: Optional[str] = None) -> List[Dict]:
        """搜索相似商品.
        
        Args:
            query: 搜索查询文本
            limit: 返回结果数量限制，默认为5
            collection_name: 要搜索的集合名称，默认为None时搜索商品基本信息集合
            
        Returns:
            相似商品列表
        """
        # 如果未指定集合，默认搜索商品基本信息集合
        if collection_name is None:
            collection_name = self.items_collection
            
        # 生成查询向量
        query_vector = self._embed_text(query)
            
        # 执行搜索
        results = self.client.search(
            collection_name=collection_name,
            query_vectors=[query_vector],
            limit=limit
        )
        
        # 处理结果
        if not results or not results[0]:
            return []
            
        # 转换结果格式
        items = []
        for hit in results[0]:
            item = hit.get("metadata", {})
            item["score"] = hit.get("score", 0.0)
            items.append(item)
            
        return items 