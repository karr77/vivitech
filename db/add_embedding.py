import json
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import pickle
import os
from tqdm import tqdm
import logging
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XianyuEmbeddingGenerator:
    """闲鱼数据Embedding生成器"""
    
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        """
        初始化Embedding生成器
        
        Args:
            model_name: 使用的预训练模型名称
                - "shibing624/text2vec-base-chinese": 轻量级，速度快
                - "moka-ai/m3e-base": 效果更好，稍大一些
                - "BAAI/bge-large-zh": 最新SOTA，但模型较大
        """
        self.model_name = model_name
        self.model = None
        self.vector_dim = None
        
        # 文本清洗正则模式
        self.clean_patterns = [
            (r'[^\w\s\u4e00-\u9fff]', ''),  # 保留中文、英文、数字、空格
            (r'\s+', ' '),  # 多个空格合并为一个
            (r'^\s+|\s+$', ''),  # 去除首尾空格
        ]
        
        # 停用词（可以根据需要扩展）
        self.stop_words = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', 
            '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', 
            '会', '着', '没有', '看', '好', '自己', '这'
        ])

    def load_model(self) -> None:
        """加载预训练模型"""
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # 获取向量维度
            test_text = "测试文本"
            test_embedding = self.model.encode(test_text)
            self.vector_dim = len(test_embedding)
            
            logger.info(f"模型加载成功，向量维度: {self.vector_dim}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 应用清洗规则
        cleaned_text = text.strip()
        for pattern, replacement in self.clean_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # 去除停用词（可选，对于embedding可能不是必须的）
        # words = cleaned_text.split()
        # cleaned_text = ' '.join([w for w in words if w not in self.stop_words])
        
        return cleaned_text

    def extract_brand_and_type(self, title: str) -> Tuple[str, str]:
        """
        从标题中提取品牌和类型
        
        Args:
            title: 商品标题
            
        Returns:
            (品牌, 类型)
        """
        words = title.split()
        brand = words[0] if len(words) > 0 else ""
        product_type = words[1] if len(words) > 1 else ""
        
        return brand, product_type

    def prepare_product_text(self, product: Dict) -> str:
        """
        为商品数据准备embedding文本，优化匹配策略
        
        Args:
            product: 商品数据字典
            
        Returns:
            拼接后的文本
        """
        components = []
        
        # 提取品牌和类型（核心匹配信息）
        brand = ""
        product_type = ""
        
        if product.get('title'):
            title = product['title']
            components.append(title)  # 原始标题
            components.append(title)  # 重复一次提高权重
            
            # 尝试提取品牌和类型
            words = title.split()
            if len(words) > 0:
                brand = words[0]
            if len(words) > 1:
                product_type = words[1]
            
            # 强化品牌和类型的权重
            if brand and product_type:
                components.append(f"{brand} {product_type}")  # 品牌+类型
                components.append(f"{brand} {product_type}")  # 再次重复增加权重
        
        # 分类信息（次要匹配信息）
        if product.get('category'):
            components.append(product['category'])
        if product.get('subcategory'):
            subcategory = product['subcategory']
            components.append(subcategory)
            # 如果商品类型为空，使用subcategory
            if not product_type:
                product_type = subcategory
                # 品牌+子类别重复以增强权重
                if brand:
                    components.append(f"{brand} {product_type}")
            
        # 描述（补充信息，权重较低）
        if product.get('description'):
            description = product['description']
            # 将描述信息截断，避免稀释关键特征
            if len(description) > 50:
                description = description[:50]
            components.append(description)
            
        # 成色（补充信息）
        if product.get('condition'):
            components.append(product['condition'])
        
        # 拼接文本
        combined_text = ' '.join(components)
        return self.clean_text(combined_text)

    def prepare_qa_text(self, qa_pair: Dict) -> Tuple[str, str]:
        """
        为问答对准备embedding文本
        
        Args:
            qa_pair: 问答对数据
            
        Returns:
            (question_text, answer_text)
        """
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        
        # 为问题添加问题类型信息并增强权重
        if qa_pair.get('question_type'):
            question_type = qa_pair['question_type']
            # 问题类型重复以增强权重
            question = f"{question_type} {question_type} {question}"
        
        return self.clean_text(question), self.clean_text(answer)

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量生成embeddings
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            embeddings数组
        """
        if not self.model:
            raise ValueError("模型未加载，请先调用load_model()")
        
        embeddings = []
        
        logger.info(f"开始生成{len(texts)}个文本的embeddings...")
        
        # 分批处理
        for i in tqdm(range(0, len(texts), batch_size), desc="生成embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, 
                                               convert_to_numpy=True,
                                               show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        # 合并所有批次
        all_embeddings = np.vstack(embeddings)
        logger.info(f"embeddings生成完成，形状: {all_embeddings.shape}")
        
        return all_embeddings

    def process_products(self, products: List[Dict]) -> Tuple[List[str], np.ndarray, List[Dict]]:
        """
        处理商品数据
        
        Args:
            products: 商品列表
            
        Returns:
            (texts, embeddings, processed_products)
        """
        logger.info(f"开始处理{len(products)}个商品...")
        
        texts = []
        processed_products = []
        
        for product in products:
            # 准备文本
            text = self.prepare_product_text(product)
            if text:  # 只处理非空文本
                texts.append(text)
                
                # 添加处理后的商品信息
                processed_product = product.copy()
                processed_product['embedding_text'] = text
                processed_products.append(processed_product)
        
        # 生成embeddings
        embeddings = self.generate_embeddings_batch(texts)
        
        logger.info(f"商品处理完成，有效商品数: {len(processed_products)}")
        return texts, embeddings, processed_products

    def process_qa_pairs(self, qa_pairs: List[Dict]) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, List[Dict]]:
        """
        处理问答对数据
        
        Args:
            qa_pairs: 问答对列表
            
        Returns:
            (question_texts, answer_texts, question_embeddings, answer_embeddings, processed_qa_pairs)
        """
        logger.info(f"开始处理{len(qa_pairs)}个问答对...")
        
        question_texts = []
        answer_texts = []
        processed_qa_pairs = []
        
        for qa_pair in qa_pairs:
            question_text, answer_text = self.prepare_qa_text(qa_pair)
            
            if question_text and answer_text:  # 只处理非空的问答对
                question_texts.append(question_text)
                answer_texts.append(answer_text)
                
                # 添加处理后的问答对信息
                processed_qa = qa_pair.copy()
                processed_qa['question_embedding_text'] = question_text
                processed_qa['answer_embedding_text'] = answer_text
                processed_qa_pairs.append(processed_qa)
        
        # 分别生成问题和答案的embeddings
        question_embeddings = self.generate_embeddings_batch(question_texts)
        answer_embeddings = self.generate_embeddings_batch(answer_texts)
        
        logger.info(f"问答对处理完成，有效问答对数: {len(processed_qa_pairs)}")
        return question_texts, answer_texts, question_embeddings, answer_embeddings, processed_qa_pairs

    def save_embeddings(self, save_dir: str = "embeddings_data"):
        """
        保存所有embedding数据
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存产品embeddings
        if hasattr(self, 'product_embeddings'):
            np.save(os.path.join(save_dir, "product_embeddings.npy"), self.product_embeddings)
            logger.info("商品embeddings已保存")
        
        # 保存问答embeddings
        if hasattr(self, 'question_embeddings'):
            np.save(os.path.join(save_dir, "question_embeddings.npy"), self.question_embeddings)
            np.save(os.path.join(save_dir, "answer_embeddings.npy"), self.answer_embeddings)
            logger.info("问答embeddings已保存")
        
        # 保存处理后的数据
        if hasattr(self, 'processed_products'):
            with open(os.path.join(save_dir, "processed_products.json"), 'w', encoding='utf-8') as f:
                json.dump(self.processed_products, f, ensure_ascii=False, indent=2)
            logger.info("处理后的商品数据已保存")
        
        if hasattr(self, 'processed_qa_pairs'):
            with open(os.path.join(save_dir, "processed_qa_pairs.json"), 'w', encoding='utf-8') as f:
                json.dump(self.processed_qa_pairs, f, ensure_ascii=False, indent=2)
            logger.info("处理后的问答数据已保存")
        
        # 保存元数据
        metadata = {
            "model_name": self.model_name,
            "vector_dimension": self.vector_dim,
            "product_count": len(getattr(self, 'processed_products', [])),
            "qa_pair_count": len(getattr(self, 'processed_qa_pairs', [])),
            "generated_at": datetime.now().isoformat()
        }
        
        with open(os.path.join(save_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"所有数据已保存到 {save_dir} 目录")

    def load_mock_data(self, file_path: str) -> Dict:
        """
        加载mock数据
        
        Args:
            file_path: mock数据文件路径
            
        Returns:
            数据字典
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"已加载mock数据: {len(data['products'])}个商品, {len(data['qa_pairs'])}个问答对")
        return data

    def process_all_data(self, mock_data_path: str, save_dir: str = "embeddings_data"):
        """
        处理所有数据的主函数
        
        Args:
            mock_data_path: mock数据文件路径
            save_dir: 保存目录
        """
        # 加载模型
        self.load_model()
        
        # 加载数据
        data = self.load_mock_data(mock_data_path)
        
        # 处理商品数据
        product_texts, self.product_embeddings, self.processed_products = self.process_products(data['products'])
        
        # 处理问答数据
        question_texts, answer_texts, self.question_embeddings, self.answer_embeddings, self.processed_qa_pairs = self.process_qa_pairs(data['qa_pairs'])
        
        # 保存所有数据
        self.save_embeddings(save_dir)
        
        # 输出统计信息
        self.print_statistics()

    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*50)
        print("📊 Embedding生成统计")
        print("="*50)
        print(f"🏷️  使用模型: {self.model_name}")
        print(f"📏 向量维度: {self.vector_dim}")
        print(f"🛍️  商品数量: {len(self.processed_products)}")
        print(f"❓ 问答对数量: {len(self.processed_qa_pairs)}")
        print(f"📦 商品embeddings形状: {self.product_embeddings.shape}")
        print(f"❓ 问题embeddings形状: {self.question_embeddings.shape}")
        print(f"💬 答案embeddings形状: {self.answer_embeddings.shape}")
        
        # 显示一些样例
        print("\n📝 样例数据:")
        if self.processed_products:
            print(f"商品样例文本: {self.processed_products[0]['embedding_text'][:100]}...")
        if self.processed_qa_pairs:
            print(f"问题样例文本: {self.processed_qa_pairs[0]['question_embedding_text']}")
            print(f"答案样例文本: {self.processed_qa_pairs[0]['answer_embedding_text']}")

    def test_similarity(self, query: str, top_k: int = 5):
        """
        测试相似度搜索，使用余弦相似度
        
        Args:
            query: 查询文本
            top_k: 返回top-k结果
        """
        if not self.model:
            logger.error("模型未加载")
            return
        
        # 生成查询向量
        query_text = self.clean_text(query)
        query_embedding = self.model.encode([query_text])
        
        # 计算与商品的相似度 - 使用余弦相似度
        if hasattr(self, 'product_embeddings'):
            # 使用余弦相似度计算
            product_similarities = cosine_similarity(
                self.product_embeddings, query_embedding
            ).flatten()
            top_product_indices = np.argsort(product_similarities)[::-1][:top_k]
            
            print(f"\n🔍 查询: '{query}'")
            print("📦 最相似的商品:")
            for i, idx in enumerate(top_product_indices, 1):
                similarity = product_similarities[idx]
                product = self.processed_products[idx]
                print(f"{i}. [{similarity:.3f}] {product['title']}")
        
        # 计算与问题的相似度 - 使用余弦相似度
        if hasattr(self, 'question_embeddings'):
            # 使用余弦相似度计算
            question_similarities = cosine_similarity(
                self.question_embeddings, query_embedding
            ).flatten()
            top_question_indices = np.argsort(question_similarities)[::-1][:top_k]
            
            print("\n❓ 最相似的问题:")
            for i, idx in enumerate(top_question_indices, 1):
                similarity = question_similarities[idx]
                qa_pair = self.processed_qa_pairs[idx]
                print(f"{i}. [{similarity:.3f}] {qa_pair['question']}")
                print(f"   答案: {qa_pair['answer']}")

def main():
    """主函数"""
    # 配置参数
    MOCK_DATA_PATH = "tests/xianyu_mock_data.json"
    SAVE_DIR = "embeddings_data"
    MODEL_NAME = "shibing624/text2vec-base-chinese"  # 可以改为其他模型
    
    # 创建生成器
    generator = XianyuEmbeddingGenerator(model_name=MODEL_NAME)
    
    try:
        # 处理所有数据
        generator.process_all_data(MOCK_DATA_PATH, SAVE_DIR)
        
        # 测试相似度搜索
        print("\n🧪 测试相似度搜索:")
        test_queries = [
            "iPhone手机",
            "便宜的衣服",
            "还在吗",
            "能便宜点吗"
        ]
        
        for query in test_queries:
            generator.test_similarity(query, top_k=3)
            print("-" * 50)
            
        print("\n✅ 所有处理完成！")
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()