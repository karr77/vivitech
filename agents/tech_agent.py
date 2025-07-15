"""
闲鱼智能客服 - 技术咨询Agent
处理与商品技术参数、规格、性能等相关的用户咨询
"""

import os
from typing import Dict, Any
from loguru import logger
from openai import OpenAI
from pymilvus import connections, utility, Collection

from .base import BaseAgent

class TechAgent(BaseAgent):
    """
    技术咨询Agent
    
    功能：回答与商品技术参数、规格、性能等相关的问题
    """
    
    def __init__(self, openai_client: OpenAI, item_info: Dict[str, Any]):
        """初始化技术咨询Agent"""
        super().__init__(item_info=item_info)
        self.openai_client = openai_client
        logger.info("技术Agent初始化完成")
        
        # 技术咨询提示词
        self.tech_prompt = """
你是闲鱼平台上的商品技术专家，专门负责解答关于商品技术规格、参数、配置、尺寸、材质等方面的问题。

【商品信息】
{item_info}

【可用型号信息】
{available_models}

【用户问题】
{user_msg}

重要提示：
- 如果用户询问的型号在"可用型号信息"中存在，一定要明确回答有这个型号
- 仔细检查"可用型号信息"中的所有型号，确保回答准确
- 优先使用数据库提供的可用型号信息，不要臆测或猜测

回复原则：
1. 极简回复，控制在1-2句话内
2. 自然口语化，像真实聊天
3. 避免过度礼貌用语，如"感谢您的..."
4. 准确回答型号、尺寸、材质等技术问题
5. 有就直说有，没有就直说没有

回复风格示例：
"这款是全新的，容量25L，可以放14寸笔记本"
"我有蓝色和黑色两种，你要哪个？"
"4QUE008型号现在有货，尺寸是42×30×13cm"
"抱歉没这个型号，我有的是4QUT7G7，要看看吗？"

返回简短、自然的回复。
"""
    
    async def process(self, user_msg: str) -> str:
        """
        处理用户消息
        
        Args:
            user_msg: 用户消息
            
        Returns:
            str: 回复内容
        """
        logger.debug(f"TechAgent处理消息 - 用户消息: '{user_msg}'")

        # 构建消息
        item_desc = self.item_info.get('desc', '')
        item_info_text = f"商品描述: {item_desc[:100]}..."
        
        if 'brand' in self.item_info:
            brand = self.item_info['brand']
            item_info_text += f"; 品牌: {brand}"
        if 'model' in self.item_info:
            model = self.item_info['model']
            item_info_text += f"; 型号: {model}"

        available_models = self._get_available_models()
        available_models_text = "可用型号: " + (", ".join(available_models) if available_models else "暂无其他型号信息")
        
        messages = [
            {"role": "system", "content": self.tech_prompt.format(
                item_info=item_info_text,
                available_models=available_models_text,
                user_msg=user_msg
            )},
            {"role": "user", "content": user_msg}
        ]
        
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            messages=messages,
            temperature=0.4,
            max_tokens=300
        )
        
        reply = response.choices[0].message.content
        
        return reply
    
    def _get_available_models(self):
        """
        从Milvus数据库获取所有可用的型号。
        """
        MILVUS_URI = "http://localhost:19530"
        MILVUS_TOKEN = "root:Milvus"
        DATABASE_NAME = "test_data"
        COLLECTION_NAME = "products"

        try:
            if "default" not in connections.list_connections():
                 connections.connect(
                    alias="default",
                    uri=MILVUS_URI,
                    token=MILVUS_TOKEN,
                    db_name=DATABASE_NAME
                )
            
            if not utility.has_collection(COLLECTION_NAME):
                logger.error(f"Milvus中不存在集合 '{COLLECTION_NAME}'")
                return []
                
            collection = Collection(COLLECTION_NAME)
            collection.load()

            results = collection.query(
                expr="model != ''",
                output_fields=["model"],
                limit=16384
            )
            
            models = set()
            for res in results:
                model_value = res.get('model')
                if model_value and isinstance(model_value, str):
                    cleaned_model = model_value.strip().upper()
                    if cleaned_model:
                        models.add(cleaned_model)

            unique_models = list(models)
            logger.info(f"从Milvus找到的所有唯一型号: {unique_models}")
            
            collection.release()
            
            return unique_models

        except Exception as e:
            logger.error(f"从 Milvus 获取可用型号时出错: {e}")
            return [] 