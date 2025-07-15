import json
import numpy as np
import time
from typing import List, Dict, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections, db
import logging
from datetime import datetime
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorSearchEngine:
    """基础向量检索引擎"""
    
    def __init__(self, 
                 uri: str = "http://43.140.216.157:18001",
                 token: str = "root:Haipro003838.",
                 db_name: str = "test_data",
                 model_name: str = "BAAI/bge-large-zh-v1.5"):
        """
        初始化向量检索引擎
        
        Args:
            uri: Milvus服务器地址
            token: 认证token
            db_name: 数据库名称
            model_name: embedding模型名称
        """
        self.uri = uri
        self.token = token
        self.db_name = db_name
        self.model_name = model_name
        
        # 集合名称
        self.product_collection_name = "xianyu_products"
        self.qa_collection_name = "xianyu_qa_pairs_pairs"
        
        # 初始化组件
        self.model = None
        self.product_collection = None
        self.qa_collection = None
        
        # 搜索统计
        self.search_stats = {
            "total_searches": 0,
            "total_time": 0,
            "avg_time": 0
        }
        
        # 初始化
        self._initialize()

    def _initialize(self):
        """初始化所有组件"""
        self._connect_to_milvus()
        self._load_embedding_model()
        self._load_collections()

    def _connect_to_milvus(self):
        """连接到Milvus数据库"""
        try:
            connections.connect(
                alias="default",
                uri=self.uri,
                token=self.token,
                secure=False
            )
            
            # 使用指定数据库
            db.using_database(self.db_name)
            logger.info(f"成功连接到Milvus数据库: {self.db_name}")
            
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise

    def _load_embedding_model(self):
        """加载embedding模型"""
        try:
            logger.info(f"正在加载embedding模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("embedding模型加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def _load_collections(self):
        """加载集合"""
        try:
            self.product_collection = Collection(self.product_collection_name)
            self.qa_collection = Collection(self.qa_collection_name)
            
            # 加载到内存
            self.product_collection.load()
            self.qa_collection.load()
            
            logger.info(f"集合加载成功:")
            logger.info(f"  - 商品集合: {self.product_collection.num_entities} 条记录")
            logger.info(f"  - 问答集合: {self.qa_collection.num_entities} 条记录")
            
        except Exception as e:
            logger.error(f"集合加载失败: {e}")
            raise

    def clean_and_encode_query(self, query: str) -> np.ndarray:
        """
        清洗查询文本并生成embedding
        
        Args:
            query: 查询文本
            
        Returns:
            查询向量
        """
        # 简单文本清洗
        cleaned_query = re.sub(r'[^\w\s\u4e00-\u9fff]', '', query.strip())
        
        # 生成embedding
        query_embedding = self.model.encode([cleaned_query])
        return query_embedding

    def search_products(self, 
                       query: str, 
                       top_k: int = 10,
                       filters: Optional[Dict] = None) -> List[Dict]:
        """
        搜索商品
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件 {"category": "数码", "price_range": [100, 1000], "location": "北京"}
            
        Returns:
            搜索结果列表
        """
        start_time = time.time()
        
        try:
            # 生成查询向量
            query_embedding = self.clean_and_encode_query(query)
            
            # 构建搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}
            }
            
            # 构建过滤表达式
            filter_expr = self._build_filter_expression(filters)
            
            # 定义输出字段
            output_fields = [
                "id", "title", "description", "brand", "category", "subcategory",
                "price", "condition", "status", "location", "views", "likes", "embedding_text"
            ]
            
            # 执行搜索
            search_results = self.product_collection.search(
                data=query_embedding,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields
            )
            
            # 处理结果
            results = []
            for hits in search_results:
                for hit in hits:
                    result = {
                        "id": hit.entity.get("id"),
                        "title": hit.entity.get("title"),
                        "description": hit.entity.get("description"),
                        "brand": hit.entity.get("brand"),
                        "category": hit.entity.get("category"),
                        "subcategory": hit.entity.get("subcategory"),
                        "price": hit.entity.get("price"),
                        "condition": hit.entity.get("condition"),
                        "status": hit.entity.get("status"),
                        "location": hit.entity.get("location"),
                        "views": hit.entity.get("views"),
                        "likes": hit.entity.get("likes"),
                        "embedding_text": hit.entity.get("embedding_text"),
                        "score": hit.score,
                        "distance": hit.distance
                    }
                    results.append(result)
            
            # 更新统计信息
            search_time = time.time() - start_time
            self._update_search_stats(search_time)
            
            logger.info(f"商品搜索完成: 查询='{query}', 结果数={len(results)}, 耗时={search_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"商品搜索失败: {e}")
            return []

    def search_qa_pairs(self, 
                       query: str, 
                       top_k: int = 10,
                       search_type: str = "question",
                       filters: Optional[Dict] = None) -> List[Dict]:
        """
        搜索问答对
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            search_type: 搜索类型 ("question", "answer", "both")
            filters: 过滤条件 {"question_type": "价格咨询", "product_id": "xxx"}
            
        Returns:
            搜索结果列表
        """
        start_time = time.time()
        
        try:
            # 生成查询向量
            query_embedding = self.clean_and_encode_query(query)
            
            # 构建搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}
            }
            
            # 构建过滤表达式
            filter_expr = self._build_qa_filter_expression(filters)
            
            # 定义输出字段
            output_fields = [
                "product_id", "question", "answer", "question_type", 
                "timestamp", "question_embedding_text", "answer_embedding_text"
            ]
            
            results = []
            
            # 根据搜索类型执行搜索
            if search_type in ["question", "both"]:
                # 搜索问题
                question_results = self.qa_collection.search(
                    data=query_embedding,
                    anns_field="question_embedding",
                    param=search_params,
                    limit=top_k,
                    expr=filter_expr,
                    output_fields=output_fields
                )
                
                for hits in question_results:
                    for hit in hits:
                        result = {
                            "product_id": hit.entity.get("product_id"),
                            "question": hit.entity.get("question"),
                            "answer": hit.entity.get("answer"),
                            "question_type": hit.entity.get("question_type"),
                            "timestamp": hit.entity.get("timestamp"),
                            "match_type": "question",
                            "score": hit.score,
                            "distance": hit.distance
                        }
                        results.append(result)
            
            if search_type in ["answer", "both"]:
                # 搜索答案
                answer_results = self.qa_collection.search(
                    data=query_embedding,
                    anns_field="answer_embedding",
                    param=search_params,
                    limit=top_k,
                    expr=filter_expr,
                    output_fields=output_fields
                )
                
                for hits in answer_results:
                    for hit in hits:
                        result = {
                            "product_id": hit.entity.get("product_id"),
                            "question": hit.entity.get("question"),
                            "answer": hit.entity.get("answer"),
                            "question_type": hit.entity.get("question_type"),
                            "timestamp": hit.entity.get("timestamp"),
                            "match_type": "answer",
                            "score": hit.score,
                            "distance": hit.distance
                        }
                        results.append(result)
            
            # 如果是both模式，需要去重并排序
            if search_type == "both":
                # 简单去重（基于product_id + question）
                seen = set()
                unique_results = []
                for result in results:
                    key = (result["product_id"], result["question"])
                    if key not in seen:
                        seen.add(key)
                        unique_results.append(result)
                
                # 按分数排序
                results = sorted(unique_results, key=lambda x: x["score"], reverse=True)[:top_k]
            
            # 更新统计信息
            search_time = time.time() - start_time
            self._update_search_stats(search_time)
            
            logger.info(f"问答搜索完成: 查询='{query}', 类型={search_type}, 结果数={len(results)}, 耗时={search_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"问答搜索失败: {e}")
            return []

    def _build_filter_expression(self, filters: Optional[Dict]) -> Optional[str]:
        """构建商品过滤表达式"""
        if not filters:
            return None
        
        expressions = []
        
        # 分类过滤
        if "category" in filters:
            expressions.append(f'category == "{filters["category"]}"')
        
        if "subcategory" in filters:
            expressions.append(f'subcategory == "{filters["subcategory"]}"')
        
        if "brand" in filters:
            expressions.append(f'brand == "{filters["brand"]}"')
        
        # 价格范围过滤
        if "price_range" in filters:
            min_price, max_price = filters["price_range"]
            expressions.append(f'price >= {min_price} and price <= {max_price}')
        
        # 地区过滤
        if "location" in filters:
            expressions.append(f'location == "{filters["location"]}"')
        
        # 状态过滤
        if "status" in filters:
            expressions.append(f'status == "{filters["status"]}"')
        
        # 成色过滤
        if "condition" in filters:
            expressions.append(f'condition == "{filters["condition"]}"')
        
        return " and ".join(expressions) if expressions else None

    def _build_qa_filter_expression(self, filters: Optional[Dict]) -> Optional[str]:
        """构建问答过滤表达式"""
        if not filters:
            return None
        
        expressions = []
        
        # 问题类型过滤
        if "question_type" in filters:
            expressions.append(f'question_type == "{filters["question_type"]}"')
        
        # 商品ID过滤
        if "product_id" in filters:
            expressions.append(f'product_id == "{filters["product_id"]}"')
        
        return " and ".join(expressions) if expressions else None

    def _update_search_stats(self, search_time: float):
        """更新搜索统计信息"""
        self.search_stats["total_searches"] += 1
        self.search_stats["total_time"] += search_time
        self.search_stats["avg_time"] = self.search_stats["total_time"] / self.search_stats["total_searches"]

    def get_search_stats(self) -> Dict:
        """获取搜索统计信息"""
        return self.search_stats.copy()

    def hybrid_search(self, 
                     query: str, 
                     top_k: int = 10,
                     include_products: bool = True,
                     include_qa: bool = True,
                     filters: Optional[Dict] = None) -> Dict:
        """
        混合搜索（商品+问答）
        
        Args:
            query: 查询文本
            top_k: 每种类型返回的结果数量
            include_products: 是否包含商品搜索
            include_qa: 是否包含问答搜索
            filters: 过滤条件
            
        Returns:
            包含多种结果的字典
        """
        results = {}
        
        if include_products:
            results["products"] = self.search_products(query, top_k, filters)
        
        if include_qa:
            results["qa_pairs"] = self.search_qa_pairs(query, top_k, "question", filters)
        
        return results

    def print_search_results(self, results: List[Dict], result_type: str = "product"):
        """格式化打印搜索结果"""
        if not results:
            print("❌ 没有找到相关结果")
            return
        
        print(f"\n🔍 搜索结果 ({result_type}):")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            if result_type == "product":
                print(f"{i}. [{result['score']:.3f}] {result['title']}")
                print(f"   品牌: {result['brand']} | 分类: {result['category']}")
                print(f"   价格: ¥{result['price']} | 成色: {result['condition']}")
                print(f"   地区: {result['location']} | 状态: {result['status']}")
                print(f"   描述: {result['description'][:100]}...")
            
            elif result_type == "qa":
                print(f"{i}. [{result['score']:.3f}] {result['question_type']}")
                print(f"   问题: {result['question']}")
                print(f"   答案: {result['answer']}")
                print(f"   匹配类型: {result['match_type']}")
            
            print("-" * 80)

def main():
    """主函数 - 测试向量搜索功能"""
    # 初始化搜索引擎
    logger.info("初始化向量搜索引擎...")
    search_engine = VectorSearchEngine()
    
    # 测试用例
    test_queries = [
        {
            "query": "iPhone手机",
            "description": "测试品牌+商品搜索"
        },
        {
            "query": "便宜的衣服",
            "description": "测试价格+分类搜索"
        },
        {
            "query": "还在吗",
            "description": "测试问答搜索"
        },
        {
            "query": "能便宜点吗",
            "description": "测试价格咨询"
        }
    ]
    
    print("\n" + "="*100)
    print("🧪 向量搜索引擎测试")
    print("="*100)
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        description = test_case["description"]
        
        print(f"\n📝 测试用例 {i}: {description}")
        print(f"🔍 查询: '{query}'")
        
        # 1. 商品搜索
        print(f"\n📦 商品搜索结果:")
        product_results = search_engine.search_products(query, top_k=3)
        if product_results:
            for j, result in enumerate(product_results, 1):
                print(f"  {j}. [{result['score']:.3f}] {result['title']}")
                print(f"     ¥{result['price']} | {result['brand']} | {result['location']}")
        else:
            print("  ❌ 无结果")
        
        # 2. 问答搜索
        print(f"\n❓ 问答搜索结果:")
        qa_results = search_engine.search_qa_pairs(query, top_k=3)
        if qa_results:
            for j, result in enumerate(qa_results, 1):
                print(f"  {j}. [{result['score']:.3f}] {result['question']}")
                print(f"     答案: {result['answer'][:50]}...")
        else:
            print("  ❌ 无结果")
        
        print("-" * 100)
    
    # 测试过滤功能
    print(f"\n🔧 过滤功能测试:")
    
    # 分类过滤
    print(f"\n1. 分类过滤测试 - 只搜索'数码'类商品:")
    filtered_results = search_engine.search_products(
        "手机", 
        top_k=3, 
        filters={"category": "数码"}
    )
    for result in filtered_results:
        print(f"   - {result['title']} (分类: {result['category']})")
    
    # 价格范围过滤
    print(f"\n2. 价格过滤测试 - 价格在500-2000元之间:")
    price_filtered_results = search_engine.search_products(
        "商品", 
        top_k=3, 
        filters={"price_range": [500, 2000]}
    )
    for result in price_filtered_results:
        print(f"   - {result['title']} (价格: ¥{result['price']})")
    
    # 地区过滤
    print(f"\n3. 地区过滤测试 - 只显示北京地区:")
    location_filtered_results = search_engine.search_products(
        "商品", 
        top_k=3, 
        filters={"location": "北京"}
    )
    for result in location_filtered_results:
        print(f"   - {result['title']} (地区: {result['location']})")
    
    # 打印统计信息
    stats = search_engine.get_search_stats()
    print(f"\n📊 搜索统计:")
    print(f"   总搜索次数: {stats['total_searches']}")
    print(f"   总耗时: {stats['total_time']:.3f}s")
    print(f"   平均耗时: {stats['avg_time']:.3f}s")
    
    print(f"\n✅ 基础向量搜索测试完成!")

if __name__ == "__main__":
    main()