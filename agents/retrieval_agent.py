"""
闲鱼智能客服 - 检索Agent
基于混合检索系统，提供相关商品信息
"""

from typing import Dict, List, Any
import os
from loguru import logger

from .base import BaseAgent
from openai import OpenAI

class RetrievalAgent(BaseAgent):
    """
    检索Agent
    
    功能：根据用户问题，从知识库中检索相关商品信息，生成回复
    """
    
    def __init__(self, item_info: Dict[str, Any], hybrid_retriever=None, openai_client: OpenAI = None):
        """
        初始化检索Agent
        
        Args:
            item_info: 当前商品信息
            hybrid_retriever: 混合检索器实例
            openai_client: OpenAI客户端
        """
        super().__init__(item_info=item_info)
        self.retriever = hybrid_retriever
        self.openai_client = openai_client
        logger.info("检索Agent初始化完成")
        
        self.synthesis_prompt = """
作为一名智能客服，你的任务是根据用户问题和检索到的相关信息，生成一个简洁、友好、自然的回复。

【用户问题】
{user_msg}

【检索到的相关信息】
{retrieved_info}

回复要求：
1. 直接回答用户问题，不要说"根据检索信息..."。
2. 如果检索信息足以回答问题，就直接利用信息回答。
3. 如果检索信息不相关或不足以回答，就礼貌地表示不确定，例如说"关于这个问题，我需要再确认一下"。
4. 保持口语化，像真人客服一样。
5. 回复内容控制在1-2句话。
"""
        
    async def process(self, user_msg: str) -> str:
        """
        处理用户消息
        
        Args:
            user_msg: 用户消息
            
        Returns:
            str: 回复内容
        """
        logger.info(f"检索Agent处理: {user_msg[:30]}...")
        
        if self.retriever is None:
            return "我们的系统正在升级中，暂时无法提供详细信息。"
        
        try:
            # 1. 执行检索
            retrieved_items = self.retriever.search(user_msg, top_k=3)
            
            # 2. 格式化检索结果
            retrieval_text = self._format_retrieval_results(retrieved_items)
            
            # 3. 使用LLM生成回复
            messages = [
                {"role": "system", "content": self.synthesis_prompt.format(
                    user_msg=user_msg,
                    retrieved_info=retrieval_text
                )},
                {"role": "user", "content": user_msg}
            ]
            
            response = self.openai_client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "gpt-4o"),
                messages=messages,
                temperature=0.4,
                max_tokens=500
            )
            
            reply = response.choices[0].message.content
            
            return reply
            
        except Exception as e:
            logger.error(f"检索Agent处理失败: {e}")
            return "这款商品非常不错。如果您有关于它的具体问题，请直接提出。"
    
    def _format_retrieval_results(self, items: List[Dict]) -> str:
        """格式化检索结果"""
        if not items:
            return "未找到相关商品信息。"
        
        result = []
        for i, item in enumerate(items):
            item_text = f"商品{i+1}: "
            if 'brand' in item and item['brand']:
                item_text += f"品牌: {item['brand']}, "
            if 'model' in item and item['model']:
                item_text += f"型号: {item['model']}, "
            if 'product_type' in item and item['product_type']:
                item_text += f"类型: {item['product_type']}, "
            if 'color' in item and item['color']:
                item_text += f"颜色: {item['color']}, "
            if 'price' in item:
                item_text += f"价格: {item['price']}元, "
            if 'key_features' in item and item['key_features']:
                item_text += f"特点: {', '.join(item['key_features'])}"
            
            item_text = item_text.rstrip(', ')
            result.append(item_text)
        
        return "\n".join(result)


