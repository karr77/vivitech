"""
闲鱼智能客服 - 混合检索Agent实现
结合向量语义搜索和BM25关键词检索，针对中文商品查询优化
"""

import json
import jieba
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from loguru import logger
import re

class XianyuHybridRetrieval:
    """
    闲鱼混合检索Agent
    
    功能特点：
    1. 向量语义检索 - 理解用户意图和商品语义
    2. BM25关键词检索 - 精确匹配品牌、型号、特征
    3. RRF融合算法 - 智能合并两种检索结果
    4. 中文优化 - 针对中文商品描述和查询优化
    """
    
    def __init__(self, 
                 vector_weight: float = 0.6,
                 bm25_weight: float = 0.4,
                 model_name: str = 'BAAI/bge-large-zh-v1.5'):
        """
        初始化检索Agent
        
        Args:
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重  
            model_name: 嵌入模型名称
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # 初始化嵌入模型
        logger.info(f"加载嵌入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # 数据存储
        self.products = []
        self.embeddings = None
        self.bm25_index = None
        self.tokenized_docs = []
        
        # 检索统计
        self.stats = {
            "vector_only": 0,
            "bm25_only": 0, 
            "hybrid": 0,
            "total_queries": 0
        }
        
        # 预编译正则表达式
        self.brand_pattern = re.compile(r'(fjallraven|jansport|dickies|doughnut|lululemon)', re.IGNORECASE)
        self.price_pattern = re.compile(r'(\d+)\s*[元块钱]?')
        self.color_pattern = re.compile(r'(黑色|白色|红色|蓝色|绿色|黄色|灰色|粉色|橙色|棕色)')
    
    def load_products(self, products_data: List[Dict]):
        """
        加载商品数据并构建索引
        
        Args:
            products_data: 商品数据列表
        """
        logger.info(f"开始加载 {len(products_data)} 个商品数据")
        
        self.products = products_data
        
        # 提取文本并生成向量
        texts = [item.get('embedding_text', '') for item in products_data]
        logger.info("生成商品向量嵌入...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # 构建BM25索引
        logger.info("构建BM25关键词索引...")
        self._build_bm25_index(texts)
        
        logger.info("数据加载完成！")
    
    def _build_bm25_index(self, texts: List[str]):
        """构建BM25索引"""
        # 对每个文本进行中文分词
        self.tokenized_docs = []
        for text in texts:
            # 使用jieba进行中文分词
            tokens = list(jieba.cut(text.lower()))
            # 过滤空字符和标点
            tokens = [token for token in tokens if token.strip() and len(token) > 1]
            self.tokenized_docs.append(tokens)
        
        # 构建BM25索引
        if self.tokenized_docs:
            self.bm25_index = BM25Okapi(self.tokenized_docs)
            logger.info(f"BM25索引构建完成，包含 {len(self.tokenized_docs)} 个文档")
    
    def _preprocess_query(self, query: str) -> Dict[str, any]:
        """
        查询预处理，提取关键信息
        
        Args:
            query: 用户查询
            
        Returns:
            Dict: 包含查询类型和提取信息的字典
        """
        query_info = {
            "original": query,
            "clean": query.lower().strip(),
            "type": "general",  # general, brand, price, color, model
            "brand": None,
            "price_range": None,
            "color": None,
            "model": None
        }
        
        # 提取品牌信息
        brand_match = self.brand_pattern.search(query)
        if brand_match:
            query_info["brand"] = brand_match.group(1).lower()
            query_info["type"] = "brand"
        
        # 提取价格信息
        price_matches = self.price_pattern.findall(query)
        if price_matches:
            prices = [int(p) for p in price_matches]
            query_info["price_range"] = (min(prices), max(prices))
            query_info["type"] = "price"
        
        # 提取颜色信息
        color_match = self.color_pattern.search(query)
        if color_match:
            query_info["color"] = color_match.group(1)
            query_info["type"] = "color"
        
        # 检测型号查询（包含数字和字母的组合）
        if re.search(r'[A-Za-z]+\d+|^\d+[A-Za-z]+', query):
            query_info["type"] = "model"
        
        return query_info
    
    def _vector_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        向量语义检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Tuple[int, float]]: (文档索引, 相似度分数)列表
        """
        if self.embeddings is None:
            return []
        
        # 生成查询向量
        query_embedding = self.model.encode([query])
        
        # 计算余弦相似度
        similarities = np.dot(query_embedding, self.embeddings.T).flatten()
        
        # 获取Top-K结果
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def _bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        BM25关键词检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Tuple[int, float]]: (文档索引, BM25分数)列表
        """
        if self.bm25_index is None:
            return []
        
        # 查询分词
        query_tokens = list(jieba.cut(query.lower()))
        query_tokens = [token for token in query_tokens if token.strip() and len(token) > 1]
        
        # BM25评分
        scores = self.bm25_index.get_scores(query_tokens)
        
        # 获取Top-K结果
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    
    def _rrf_fusion(self, 
                   vector_results: List[Tuple[int, float]], 
                   bm25_results: List[Tuple[int, float]], 
                   k: int = 60) -> List[Tuple[int, float]]:
        """
        使用RRF（Reciprocal Rank Fusion）算法融合检索结果
        
        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果
            k: RRF参数
            
        Returns:
            List[Tuple[int, float]]: 融合后的结果
        """
        rrf_scores = {}
        
        # 处理向量检索结果
        for rank, (doc_idx, score) in enumerate(vector_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + \
                                self.vector_weight / (k + rank + 1)
        
        # 处理BM25检索结果
        for rank, (doc_idx, score) in enumerate(bm25_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + \
                                self.bm25_weight / (k + rank + 1)
        
        # 排序并返回
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_idx, score) for doc_idx, score in sorted_results]
    
    def _adaptive_retrieval_strategy(self, query_info: Dict[str, any]) -> str:
        """
        根据查询类型确定检索策略
        
        Args:
            query_info: 查询信息
            
        Returns:
            str: 检索策略 ("vector", "bm25", "hybrid")
        """
        query_type = query_info["type"]
        
        # 型号查询优先使用BM25
        if query_type == "model":
            return "bm25"
        
        # 品牌、颜色查询使用混合检索
        elif query_type in ["brand", "color", "price"]:
            return "hybrid"
        
        # 一般性查询优先使用向量检索
        else:
            return "vector"
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        执行混合检索
        
        Args:
            query: 用户查询
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 检索到的商品列表
        """
        if not query.strip():
            return []
        
        self.stats["total_queries"] += 1
        
        # 查询预处理
        query_info = self._preprocess_query(query)
        
        # 确定检索策略
        strategy = self._adaptive_retrieval_strategy(query_info)
        
        logger.info(f"查询: '{query}' | 类型: {query_info['type']} | 策略: {strategy}")
        
        # 执行检索
        if strategy == "vector":
            self.stats["vector_only"] += 1
            vector_results = self._vector_search(query, top_k * 2)
            final_results = vector_results[:top_k]
            
        elif strategy == "bm25":
            self.stats["bm25_only"] += 1
            bm25_results = self._bm25_search(query, top_k * 2)
            final_results = bm25_results[:top_k]
            
        else:  # hybrid
            self.stats["hybrid"] += 1
            vector_results = self._vector_search(query, top_k * 2)
            bm25_results = self._bm25_search(query, top_k * 2)
            final_results = self._rrf_fusion(vector_results, bm25_results)[:top_k]
        
        # 返回商品信息
        products = []
        for doc_idx, score in final_results:
            if 0 <= doc_idx < len(self.products):
                product = self.products[doc_idx].copy()
                product["relevance_score"] = round(score, 4)
                products.append(product)
        
        return products
    
    def get_statistics(self) -> Dict[str, any]:
        """获取检索统计信息"""
        total = self.stats["total_queries"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "vector_ratio": round(self.stats["vector_only"] / total, 3),
            "bm25_ratio": round(self.stats["bm25_only"] / total, 3),
            "hybrid_ratio": round(self.stats["hybrid"] / total, 3)
        }
    
    def update_weights(self, vector_weight: float, bm25_weight: float):
        """动态调整检索权重"""
        total = vector_weight + bm25_weight
        self.vector_weight = vector_weight / total
        self.bm25_weight = bm25_weight / total
        logger.info(f"权重更新: 向量={self.vector_weight:.2f}, BM25={self.bm25_weight:.2f}")


# 使用示例
if __name__ == "__main__":
    # 从JSON文件加载商品数据
    import json
    
    try:
        # 加载清洗后的商品数据
        with open('data/cleaned_product.json', 'r', encoding='utf-8') as f:
            products_data = json.load(f)
        print(f"成功加载了{len(products_data)}个商品数据")
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        # 如果加载失败，使用示例数据
        products_data = [
            {
                "id": "939866773100",
                "brand": "Fjallraven",
                "model": "23799",
                "product_type": "双肩包",
                "color": "蓝色",
                "price": 135.0,
                "key_features": ["帆布", "防泼水"],
                "embedding_text": "品牌：Fjallraven 型号：23799 类型：双肩包 颜色：蓝色 价格：135元 特征：帆布 防泼水"
            }
        ]
    
    # 初始化检索器
    retriever = XianyuHybridRetrieval()
    
    # 加载数据
    retriever.load_products(products_data)
    
    # 执行检索示例
    test_queries = [
        "想要一个蓝色的北极狐背包",
        "有没有JanSport的麂皮双肩包",
        "100元以内的黑色背包",
        "4QUT7H6型号的包包",
        "北极狐的深蓝色看看"
    ]
    
    print("\n===== 检索测试 =====")
    for query in test_queries:
        print(f"\n查询: '{query}'")
        results = retriever.search(query, top_k=3)
        
        if results:
            print(f"找到 {len(results)} 个相关商品:")
            for i, product in enumerate(results):
                print(f"{i+1}. {product['brand']} {product['model']} - {product['product_type']} "
                      f"({product['price']}元) 相关度:{product['relevance_score']}")
        else:
            print("未找到相关商品")
    
    # 查看统计
    print("\n===== 检索统计 =====")
    stats = retriever.get_statistics()
    print(stats)