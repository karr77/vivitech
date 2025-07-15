#!/usr/bin/env python3
"""
BM25关键词搜索引擎
支持中文分词和关键词检索
"""

import json
import jieba
import pickle
import os
import re
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import numpy as np
import logging
from datetime import datetime
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChineseTokenizer:
    """中文分词器"""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
        # 配置jieba用户词典（可选）
        self._setup_jieba()
    
    def _load_stop_words(self) -> set:
        """加载中文停用词表"""
        # 基础停用词
        basic_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', 
            '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', 
            '会', '着', '没有', '看', '好', '自己', '这', '个', '来', '用',
            '大', '里', '为', '子', '中', '与', '从', '地', '他', '时',
            '部', '将', '成', '被', '而', '所', '以', '及', '其', '对',
            '可以', '可能', '应该', '如果', '因为', '所以', '但是', '然后',
            '还有', '或者', '以及', '根据', '通过', '由于', '关于'
        }
        return basic_stop_words
    
    def _setup_jieba(self):
        """配置jieba分词器"""
        # 添加自定义词典（针对电商领域）
        custom_words = [
            '闲鱼', '二手', '全新', '9成新', '8成新', '7成新', '99新', '95新',
            'iPhone', 'iPad', 'MacBook', 'AirPods', 'iPhone15', 'iPhone14',
            '华为', '小米', '魅族', 'OPPO', 'vivo', '一加', '荣耀',
            '包邮', '急转', '学生党', '搬家', '闲置', '低价', '正品',
            '面交', '当面交易', '支持退换', '质量保证'
        ]
        
        for word in custom_words:
            jieba.add_word(word)
        
        # 强制载入jieba词典
        jieba.initialize()
        logger.info("jieba分词器初始化完成")
    
    def tokenize(self, text: str) -> List[str]:
        """
        分词并清理
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        if not text or not isinstance(text, str):
            return []
        
        # 清理文本
        text = self._clean_text(text)
        
        # 使用jieba分词
        tokens = jieba.lcut(text)
        
        # 过滤停用词和短词
        filtered_tokens = []
        for token in tokens:
            token = token.strip()
            if (len(token) >= 1 and 
                token not in self.stop_words and 
                not self._is_punctuation(token) and
                not self._is_number_only(token)):
                filtered_tokens.append(token.lower())
        
        return filtered_tokens
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 保留中文、英文、数字、常用标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff\-]', ' ', text)
        return text.strip()
    
    def _is_punctuation(self, token: str) -> bool:
        """判断是否为标点符号"""
        return re.match(r'^[^\w\u4e00-\u9fff]+$', token) is not None
    
    def _is_number_only(self, token: str) -> bool:
        """判断是否为纯数字"""
        return token.isdigit()

class BM25SearchEngine:
    """BM25关键词搜索引擎"""
    
    def __init__(self, 
                 bm25_type: str = "BM25Okapi",
                 k1: float = 1.5, 
                 b: float = 0.75):
        """
        初始化BM25搜索引擎
        
        Args:
            bm25_type: BM25算法类型 ("BM25Okapi", "BM25L", "BM25Plus")
            k1: BM25参数k1
            b: BM25参数b
        """
        self.tokenizer = ChineseTokenizer()
        self.bm25_type = bm25_type
        self.k1 = k1
        self.b = b
        
        # 索引数据
        self.product_bm25 = None
        self.qa_bm25 = None
        self.product_docs = []
        self.qa_docs = []
        self.product_metadata = []
        self.qa_metadata = []
        
        logger.info(f"BM25搜索引擎初始化完成，算法类型: {bm25_type}")
    
    def _get_bm25_class(self):
        """获取BM25算法类"""
        if self.bm25_type == "BM25L":
            return BM25L
        elif self.bm25_type == "BM25Plus":
            return BM25Plus
        else:
            return BM25Okapi
    
    def load_data(self, data_file: str = "embeddings_data/processed_products.json",
                  qa_file: str = "embeddings_data/processed_qa_pairs.json"):
        """
        加载数据
        
        Args:
            data_file: 商品数据文件路径
            qa_file: 问答数据文件路径
        """
        logger.info("开始加载数据...")
        
        # 加载商品数据
        with open(data_file, 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        # 加载问答数据
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        logger.info(f"加载完成: {len(products)}个商品, {len(qa_pairs)}个问答对")
        
        # 构建索引
        self._build_product_index(products)
        self._build_qa_index(qa_pairs)
    
    def _build_product_index(self, products: List[Dict]):
        """构建商品BM25索引"""
        logger.info("构建商品BM25索引...")
        
        self.product_docs = []
        self.product_metadata = []
        
        for product in products:
            # 组合文本用于检索
            searchable_text = self._prepare_product_search_text(product)
            
            # 分词
            tokens = self.tokenizer.tokenize(searchable_text)
            self.product_docs.append(tokens)
            
            # 保存元数据
            self.product_metadata.append(product)
        
        # 创建BM25索引
        bm25_class = self._get_bm25_class()
        if self.bm25_type == "BM25Okapi":
            self.product_bm25 = bm25_class(self.product_docs, k1=self.k1, b=self.b)
        else:
            self.product_bm25 = bm25_class(self.product_docs)
        
        logger.info(f"商品BM25索引构建完成，文档数量: {len(self.product_docs)}")
    
    def _build_qa_index(self, qa_pairs: List[Dict]):
        """构建问答BM25索引"""
        logger.info("构建问答BM25索引...")
        
        self.qa_docs = []
        self.qa_metadata = []
        
        for qa in qa_pairs:
            # 组合问题和答案用于检索
            searchable_text = f"{qa.get('question', '')} {qa.get('answer', '')}"
            
            # 分词
            tokens = self.tokenizer.tokenize(searchable_text)
            self.qa_docs.append(tokens)
            
            # 保存元数据
            self.qa_metadata.append(qa)
        
        # 创建BM25索引
        bm25_class = self._get_bm25_class()
        if self.bm25_type == "BM25Okapi":
            self.qa_bm25 = bm25_class(self.qa_docs, k1=self.k1, b=self.b)
        else:
            self.qa_bm25 = bm25_class(self.qa_docs)
        
        logger.info(f"问答BM25索引构建完成，文档数量: {len(self.qa_docs)}")
    
    def _prepare_product_search_text(self, product: Dict) -> str:
        """准备商品搜索文本"""
        components = []
        
        # 标题（最重要，重复以增加权重）
        if product.get('title'):
            title = product['title']
            components.extend([title, title])
        
        # 品牌和型号
        if product.get('brand'):
            components.append(product['brand'])
        if product.get('model'):
            components.append(product['model'])
        
        # 分类信息
        if product.get('category'):
            components.append(product['category'])
        if product.get('subcategory'):
            components.append(product['subcategory'])
        
        # 描述（截取重要部分）
        if product.get('description'):
            desc = product['description'][:100]  # 限制长度
            components.append(desc)
        
        # 成色和状态
        if product.get('condition'):
            components.append(product['condition'])
        
        return ' '.join(components)
    
    def search_products(self, 
                       query: str, 
                       top_k: int = 10,
                       filters: Optional[Dict] = None) -> List[Dict]:
        """
        搜索商品
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        if not self.product_bm25:
            raise ValueError("商品索引未构建，请先调用load_data()")
        
        # 查询分词
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []
        
        # BM25搜索
        scores = self.product_bm25.get_scores(query_tokens)
        
        # 获取top_k结果
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            
            score = scores[idx]
            if score <= 0:  # 跳过无关结果
                continue
            
            product = self.product_metadata[idx].copy()
            
            # 应用过滤条件
            if filters and not self._apply_product_filters(product, filters):
                continue
            
            product['score'] = float(score)
            product['search_type'] = 'bm25'
            results.append(product)
        
        return results
    
    def search_qa_pairs(self, 
                       query: str, 
                       top_k: int = 10,
                       filters: Optional[Dict] = None) -> List[Dict]:
        """
        搜索问答对
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        if not self.qa_bm25:
            raise ValueError("问答索引未构建，请先调用load_data()")
        
        # 查询分词
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []
        
        # BM25搜索
        scores = self.qa_bm25.get_scores(query_tokens)
        
        # 获取top_k结果
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            
            score = scores[idx]
            if score <= 0:  # 跳过无关结果
                continue
            
            qa = self.qa_metadata[idx].copy()
            
            # 应用过滤条件
            if filters and not self._apply_qa_filters(qa, filters):
                continue
            
            qa['score'] = float(score)
            qa['search_type'] = 'bm25'
            results.append(qa)
        
        return results
    
    def _apply_product_filters(self, product: Dict, filters: Dict) -> bool:
        """应用商品过滤条件"""
        # 分类过滤
        if 'category' in filters:
            if product.get('category') != filters['category']:
                return False
        
        if 'subcategory' in filters:
            if product.get('subcategory') != filters['subcategory']:
                return False
        
        # 品牌过滤
        if 'brand' in filters:
            if product.get('brand') != filters['brand']:
                return False
        
        # 价格范围过滤
        if 'price_range' in filters:
            min_price, max_price = filters['price_range']
            price = product.get('price', 0)
            if not (min_price <= price <= max_price):
                return False
        
        # 地区过滤
        if 'location' in filters:
            if product.get('location') != filters['location']:
                return False
        
        # 状态过滤
        if 'status' in filters:
            if product.get('status') != filters['status']:
                return False
        
        return True
    
    def _apply_qa_filters(self, qa: Dict, filters: Dict) -> bool:
        """应用问答过滤条件"""
        # 问题类型过滤
        if 'question_type' in filters:
            if qa.get('question_type') != filters['question_type']:
                return False
        
        return True
    
    def save_index(self, save_dir: str = "bm25_index"):
        """保存BM25索引"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存商品索引
        if self.product_bm25:
            with open(os.path.join(save_dir, "product_bm25.pkl"), 'wb') as f:
                pickle.dump(self.product_bm25, f)
            
            with open(os.path.join(save_dir, "product_docs.pkl"), 'wb') as f:
                pickle.dump(self.product_docs, f)
            
            with open(os.path.join(save_dir, "product_metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(self.product_metadata, f, ensure_ascii=False, indent=2)
        
        # 保存问答索引
        if self.qa_bm25:
            with open(os.path.join(save_dir, "qa_bm25.pkl"), 'wb') as f:
                pickle.dump(self.qa_bm25, f)
            
            with open(os.path.join(save_dir, "qa_docs.pkl"), 'wb') as f:
                pickle.dump(self.qa_docs, f)
            
            with open(os.path.join(save_dir, "qa_metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(self.qa_metadata, f, ensure_ascii=False, indent=2)
        
        # 保存配置
        config = {
            "bm25_type": self.bm25_type,
            "k1": self.k1,
            "b": self.b,
            "created_at": datetime.now().isoformat()
        }
        with open(os.path.join(save_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"BM25索引已保存到 {save_dir}")
    
    def load_index(self, save_dir: str = "bm25_index"):
        """加载已保存的BM25索引"""
        logger.info(f"从 {save_dir} 加载BM25索引...")
        
        # 加载配置
        with open(os.path.join(save_dir, "config.json"), 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.bm25_type = config["bm25_type"]
        self.k1 = config["k1"]
        self.b = config["b"]
        
        # 加载商品索引
        try:
            with open(os.path.join(save_dir, "product_bm25.pkl"), 'rb') as f:
                self.product_bm25 = pickle.load(f)
            
            with open(os.path.join(save_dir, "product_docs.pkl"), 'rb') as f:
                self.product_docs = pickle.load(f)
            
            with open(os.path.join(save_dir, "product_metadata.json"), 'r', encoding='utf-8') as f:
                self.product_metadata = json.load(f)
            
            logger.info(f"商品索引加载完成: {len(self.product_docs)} 个文档")
        except FileNotFoundError:
            logger.warning("未找到商品索引文件")
        
        # 加载问答索引
        try:
            with open(os.path.join(save_dir, "qa_bm25.pkl"), 'rb') as f:
                self.qa_bm25 = pickle.load(f)
            
            with open(os.path.join(save_dir, "qa_docs.pkl"), 'rb') as f:
                self.qa_docs = pickle.load(f)
            
            with open(os.path.join(save_dir, "qa_metadata.json"), 'r', encoding='utf-8') as f:
                self.qa_metadata = json.load(f)
            
            logger.info(f"问答索引加载完成: {len(self.qa_docs)} 个文档")
        except FileNotFoundError:
            logger.warning("未找到问答索引文件")
    
    def analyze_query(self, query: str) -> Dict:
        """分析查询，返回分词结果和统计信息"""
        tokens = self.tokenizer.tokenize(query)
        
        analysis = {
            "original_query": query,
            "tokens": tokens,
            "token_count": len(tokens),
            "unique_tokens": len(set(tokens))
        }
        
        return analysis
    
    def get_index_stats(self) -> Dict:
        """获取索引统计信息"""
        stats = {
            "bm25_type": self.bm25_type,
            "parameters": {"k1": self.k1, "b": self.b},
            "product_index": {
                "document_count": len(self.product_docs) if self.product_docs else 0,
                "avg_doc_length": np.mean([len(doc) for doc in self.product_docs]) if self.product_docs else 0
            },
            "qa_index": {
                "document_count": len(self.qa_docs) if self.qa_docs else 0,
                "avg_doc_length": np.mean([len(doc) for doc in self.qa_docs]) if self.qa_docs else 0
            }
        }
        
        return stats

def main():
    """主函数 - 测试BM25搜索引擎"""
    print("🔍 BM25关键词搜索引擎测试")
    print("="*50)
    
    # 创建搜索引擎
    search_engine = BM25SearchEngine(bm25_type="BM25Okapi")
    
    try:
        # 加载数据并构建索引
        search_engine.load_data()
        
        # 保存索引
        search_engine.save_index()
        
        # 输出统计信息
        stats = search_engine.get_index_stats()
        print(f"\n📊 索引统计:")
        print(f"算法类型: {stats['bm25_type']}")
        print(f"商品文档数: {stats['product_index']['document_count']}")
        print(f"问答文档数: {stats['qa_index']['document_count']}")
        
        # 测试搜索
        test_queries = [
            "iPhone手机",
            "便宜的衣服",
            "还在吗",
            "能便宜点吗",
            "华为手机",
            "Nike鞋子"
        ]
        
        print(f"\n🧪 测试搜索:")
        for query in test_queries:
            print(f"\n查询: '{query}'")
            
            # 分析查询
            analysis = search_engine.analyze_query(query)
            print(f"分词结果: {analysis['tokens']}")
            
            # 搜索商品
            products = search_engine.search_products(query, top_k=3)
            print(f"商品结果 ({len(products)} 个):")
            for i, product in enumerate(products, 1):
                print(f"  {i}. [{product['score']:.3f}] {product['title']}")
            
            # 搜索问答
            qa_pairs = search_engine.search_qa_pairs(query, top_k=2)
            print(f"问答结果 ({len(qa_pairs)} 个):")
            for i, qa in enumerate(qa_pairs, 1):
                print(f"  {i}. [{qa['score']:.3f}] {qa['question']}")
            
            print("-" * 40)
        
        print(f"\n✅ BM25搜索引擎测试完成!")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()