"""
闲鱼智能客服 - 闲聊Agent
处理一般性聊天、问候和未分类的用户消息
"""

import os
from typing import Dict, Any
from loguru import logger
from openai import OpenAI

from .base import BaseAgent

class ChatAgent(BaseAgent):
    """
    闲聊Agent
    
    功能：处理一般性聊天、问候和未分类的用户消息
    作为对话兜底，确保所有用户消息都能得到回复
    """
    
    def __init__(self, openai_client: OpenAI, item_info: Dict[str, Any]):
        """初始化闲聊Agent"""
        super().__init__(item_info=item_info)
        self.openai_client = openai_client
        logger.info("闲聊Agent初始化完成")
        
        # 闲聊提示词
        self.chat_prompt = """
你是闲鱼平台上的智能客服，负责日常交流。

【商品信息】
{item_info}

【用户消息】
{user_msg}

回复原则：
1. 极简回复，控制在1-2句话内
2. 自然口语化，像真实聊天
3. 避免过度礼貌用语，如"感谢您的..."
4. 不要重复描述商品，只在必要时简单提及
5. 轻松随意的语气，使用简短句子

回复风格示例：
"有啊，要吗？"
"69元，今天下单可以马上发"
"要不要看看更多照片？"
"可以讲价，说个数呗"

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
        try:
            # 格式化商品信息
            item_desc = self.item_info.get('desc', '')
            item_price = self.item_info.get('soldPrice', '未知')
            item_info_text = f"商品描述: {item_desc}; 价格: {item_price}元"
            
            # 生成回复
            messages = [
                {"role": "system", "content": self.chat_prompt.format(
                    item_info=item_info_text,
                    user_msg=user_msg
                )},
                {"role": "user", "content": user_msg}
            ]
            
            # 使用较高的temperature使回复更自然
            response = self.openai_client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "gpt-4o"),
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )
            
            reply = response.choices[0].message.content
            
            return reply
            
        except Exception as e:
            logger.error(f"闲聊Agent处理失败: {e}")
            return "您好！这款商品很不错，有什么可以帮您的吗？" 