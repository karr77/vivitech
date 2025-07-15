"""
闲鱼智能客服 - 订单Agent
处理订单、发货、物流等相关的用户咨询
"""

import os
from typing import Dict, Any
from loguru import logger
from openai import OpenAI

from .base import BaseAgent

class OrderAgent(BaseAgent):
    """
    订单Agent
    
    功能：回答与订单、发货、物流等相关的问题
    """
    
    def __init__(self, openai_client: OpenAI, item_info: Dict[str, Any]):
        """初始化订单Agent"""
        super().__init__(item_info=item_info)
        self.openai_client = openai_client
        logger.info("订单Agent初始化完成")
        
        # 订单提示词
        self.order_prompt = """
你是闲鱼平台上的订单客服专家，负责解答用户关于订单、发货、物流等方面的问题。

【商品信息】
{item_info}

【用户问题】
{user_msg}

请按照以下原则回答用户的订单相关问题：
1. 发货时间：保证下单后24小时内发货
2. 物流信息：提醒用户通过闲鱼平台查看最新物流状态
3. 退换货政策：支持7天无理由退换，但商品需保持原状态
4. 订单查询：指导用户在"我的订单"中查看详情
5. 催发货：耐心解释发货流程，承诺尽快处理

请用温和友好的语气回答，简明扼要，不要超过3句话。回答要专业但不机械，关注用户需求。
"""
    
    async def process(self, user_msg: str) -> str:
        """
        处理用户消息
        
        Args:
            user_msg: 用户消息
            
        Returns:
            str: 回复内容
        """
        logger.debug(f"OrderAgent处理消息 - 用户消息: '{user_msg}'")
            
        try:
            # 格式化商品信息
            item_desc = self.item_info.get('desc', '')
            item_info_text = f"商品描述: {item_desc}"
            
            # 生成回复
            messages = [
                {"role": "system", "content": self.order_prompt.format(
                    item_info=item_info_text,
                    user_msg=user_msg
                )},
                {"role": "user", "content": user_msg}
            ]
            
            response = self.openai_client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "gpt-4o"),
                messages=messages,
                temperature=0.3,
                max_tokens=250
            )
            
            reply = response.choices[0].message.content
            
            return reply
            
        except Exception as e:
            logger.error(f"订单Agent处理失败: {e}")
            return "下单后24小时内发货，支持7天无理由退换，请放心购买。" 