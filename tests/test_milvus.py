from dataclasses import dataclass
from typing import List, Dict, Tuple
import re

@dataclass
class MultiLevelEmbedding:
    """多层次embedding结构"""
    structured_text: str      # 结构化属性文本
    semantic_text: str        # 语义扩展文本  
    description_text: str     # 清洗后描述文本
    search_keywords: List[str] # 搜索关键词列表

class EnhancedProductEmbedding:
    def __init__(self):
        # 扩展同义词词典
        self.synonym_dict = {
            # 品牌同义词
            'fjallraven': ['北极狐', '狐狸', 'kanken', '瑞典'],
            'jansport': ['杰斯伯', '杰思博', 'js', '美国', '学生包'],
            'dickies': ['工装', '休闲包'],
            'doughnut': ['甜甜圈', '港风', '时尚包'],
            
            # 功能同义词
            '防水': ['防泼水', '防雨', '户外'],
            '学生': ['书包', '校园', '上学', '读书'],
            '商务': ['办公', '电脑包', '职场'],
            '旅行': ['出行', '户外', '探险', '旅游'],
            
            # 尺寸同义词
            '小': ['mini', '迷你', '7L', '小号'],
            '中': ['中等', '标准', '16L', '中号'],
            '大': ['大容量', '20L', '25L', '大号'],
            
            # 价格同义词
            '便宜': ['低价', '特价', '实惠', '划算'],
            '贵': ['高端', '奢华', 'premium'],
        }
        
        # 查询意图分类
        self.intent_patterns = {
            'brand_query': [r'(北极狐|fjallraven|jansport|dickies)', '品牌导向'],
            'price_query': [r'(\d+元|便宜|贵|实惠|划算)', '价格导向'],
            'function_query': [r'(防水|学生|商务|旅行|户外)', '功能导向'],
            'size_query': [r'(大|中|小|容量|\d+L)', '尺寸导向'],
            'color_query': [r'(黑色|白色|红色|蓝色|绿色|黄色|粉色)', '颜色导向'],
        }

    def generate_multi_level_embedding(self, product: Dict) -> MultiLevelEmbedding:
        """生成多层次embedding"""
        
        # 1. 结构化属性文本（精确匹配用）
        structured_parts = []
        if product['brand'] != "未知品牌":
            structured_parts.append(f"BRAND_{product['brand']}")
        if product['model']:
            structured_parts.append(f"MODEL_{product['model']}")
        structured_parts.append(f"TYPE_{product['product_type']}")
        if product['color']:
            structured_parts.append(f"COLOR_{product['color']}")
        if product['size']:
            structured_parts.append(f"SIZE_{product['size']}")
        if product['capacity']:
            structured_parts.append(f"CAPACITY_{product['capacity']}")
        
        # 价格分级
        price = product['price']
        if price < 80:
            structured_parts.append("PRICE_LOW")
        elif price < 150:
            structured_parts.append("PRICE_MID")
        else:
            structured_parts.append("PRICE_HIGH")
            
        structured_text = " ".join(structured_parts)
        
        # 2. 语义扩展文本（语义匹配用）
        semantic_parts = []
        brand_lower = product['brand'].lower()
        if brand_lower in self.synonym_dict:
            semantic_parts.extend(self.synonym_dict[brand_lower])
            
        # 功能语义扩展
        for feature in product['key_features']:
            if feature in self.synonym_dict:
                semantic_parts.extend(self.synonym_dict[feature])
                
        # 用途推断
        if '学生' in product['embedding_text'] or '书包' in product['embedding_text']:
            semantic_parts.extend(['校园', '青春', '上学'])
        if '商务' in product['embedding_text'] or '电脑' in product['embedding_text']:
            semantic_parts.extend(['职场', '办公', '专业'])
        if '户外' in product['embedding_text'] or '旅行' in product['embedding_text']:
            semantic_parts.extend(['探险', '出行', '运动'])
            
        semantic_text = " ".join(set(semantic_parts))  # 去重
        
        # 3. 清洗后描述文本（上下文理解用）
        description_text = self._clean_description_advanced(product['embedding_text'])
        
        # 4. 搜索关键词（快速检索用）
        search_keywords = []
        search_keywords.append(product['brand'])
        if product['model']:
            search_keywords.append(product['model'])
        search_keywords.append(product['product_type'])
        if product['color']:
            search_keywords.append(product['color'])
        search_keywords.extend(product['key_features'])
        
        return MultiLevelEmbedding(
            structured_text=structured_text,
            semantic_text=semantic_text,
            description_text=description_text,
            search_keywords=search_keywords
        )

    def _clean_description_advanced(self, text: str) -> str:
        """高级描述清洗"""
        # 移除营销词汇
        marketing_patterns = [
            r'包邮.*?[！!]*',
            r'\[号外\].*?\[号外\]',
            r'假一.*?十',
            r'支持.*?验货',
            r'得物.*?',
            r'保证.*?正品',
        ]
        
        cleaned = text
        for pattern in marketing_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            
        # 提取核心信息
        sentences = re.split(r'[。！\n]', cleaned)
        core_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # 保留包含产品信息的句子
            if any(keyword in sentence for keyword in ['材质', '尺寸', '容量', '特点', '功能']):
                core_sentences.append(sentence)
            elif len(sentence) < 50 and any(char in sentence for char in '防水轻便'):
                core_sentences.append(sentence)
                
        return ' '.join(core_sentences[:3])  # 最多保留3个核心句子

    def create_composite_embedding_text(self, multi_embedding: MultiLevelEmbedding) -> str:
        """创建组合embedding文本"""
        # 按重要性组合不同层次的文本
        composite_parts = [
            f"[STRUCT]{multi_embedding.structured_text}[/STRUCT]",
            f"[SEMANTIC]{multi_embedding.semantic_text}[/SEMANTIC]", 
            f"[DESC]{multi_embedding.description_text}[/DESC]"
        ]
        
        return " ".join(composite_parts)

    def analyze_query_intent(self, query: str) -> Dict[str, float]:
        """分析查询意图"""
        intent_scores = {}
        
        for intent, (pattern, description) in self.intent_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            score = len(matches) / len(query.split()) if matches else 0
            intent_scores[intent] = score
            
        return intent_scores

    def optimize_query_for_search(self, query: str) -> str:
        """优化查询文本"""
        # 分析意图
        intents = self.analyze_query_intent(query)
        dominant_intent = max(intents.items(), key=lambda x: x[1])[0] if intents else None
        
        # 根据主要意图调整查询权重
        optimized_parts = []
        
        if dominant_intent == 'brand_query':
            # 品牌查询：强化品牌匹配
            brand_matches = re.findall(r'(北极狐|fjallraven|jansport|dickies)', query, re.IGNORECASE)
            for brand in brand_matches:
                optimized_parts.append(f"[STRUCT]BRAND_{brand.upper()}[/STRUCT]")
                
        elif dominant_intent == 'price_query':
            # 价格查询：添加价格分级
            if any(word in query for word in ['便宜', '低价', '实惠']):
                optimized_parts.append("[STRUCT]PRICE_LOW[/STRUCT]")
            elif any(word in query for word in ['贵', '高端']):
                optimized_parts.append("[STRUCT]PRICE_HIGH[/STRUCT]")
                
        # 添加语义扩展
        semantic_expansions = []
        for word in query.split():
            if word in self.synonym_dict:
                semantic_expansions.extend(self.synonym_dict[word])
        
        if semantic_expansions:
            optimized_parts.append(f"[SEMANTIC]{' '.join(semantic_expansions)}[/SEMANTIC]")
            
        # 原始查询
        optimized_parts.append(f"[DESC]{query}[/DESC]")
        
        return " ".join(optimized_parts)

# 使用示例
def demonstrate_enhanced_embedding():
    """演示增强版embedding效果"""
    
    # 示例产品数据
    sample_product = {
        'id': '939866773100',
        'brand': 'Fjallraven',
        'model': '23799',
        'product_type': '双肩包',
        'color': '深蓝色',
        'size': '中号',
        'capacity': '16L',
        'price': 135.0,
        'key_features': ['防泼水', '帆布', '全新'],
        'embedding_text': '品牌：Fjallraven 型号：23799 类型：双肩包 颜色：深蓝色 尺寸：中号 价格：135元 特征：防泼水 帆布 全新'
    }
    
    enhancer = EnhancedProductEmbedding()
    
    # 生成多层次embedding
    multi_embedding = enhancer.generate_multi_level_embedding(sample_product)
    
    print("=== 增强版Embedding结果 ===")
    print(f"结构化文本: {multi_embedding.structured_text}")
    print(f"语义文本: {multi_embedding.semantic_text}")
    print(f"描述文本: {multi_embedding.description_text}")
    print(f"搜索关键词: {multi_embedding.search_keywords}")
    
    # 生成最终embedding文本
    final_text = enhancer.create_composite_embedding_text(multi_embedding)
    print(f"\n最终Embedding文本:\n{final_text}")
    
    # 查询优化示例
    test_queries = [
        "便宜的北极狐背包",
        "学生用的中号书包", 
        "防水户外包"
    ]
    
    print("\n=== 查询优化示例 ===")
    for query in test_queries:
        optimized = enhancer.optimize_query_for_search(query)
        print(f"原查询: {query}")
        print(f"优化后: {optimized}")
        print()

if __name__ == "__main__":
    demonstrate_enhanced_embedding()