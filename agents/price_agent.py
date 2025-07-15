"""
闲鱼智能客服 - 价格Agent
处理与价格、议价相关的用户交互
"""

import os
from typing import Dict, Any
from loguru import logger
from openai import OpenAI

from .base import BaseAgent

class PriceAgent(BaseAgent):
    """
    价格Agent
    
    功能：专门处理与价格相关的问题，如议价、价格咨询等
    """
    
    def __init__(self, openai_client: OpenAI, item_info: Dict[str, Any]):
        """初始化价格Agent"""
        super().__init__(item_info=item_info)
        self.openai_client = openai_client
        logger.info("价格Agent初始化完成")
        
        # 价格提示词
        self.price_prompt = """
你是闲鱼平台上一位【拥有一定议价权限】的智能客服。

【你的核心规则】
1.  **识别意图**：判断用户是"问价"还是"议价"。
    -   **问价**：用户只问价格，如"多少钱？"。
    -   **议价**：用户希望降价，如"便宜点？"或直接出价。
2.  **严格按规则回应**：
    -   **如果用户是"问价"**：你【只能】报出商品原价，并可附带一句商品优点。例如："102元，全新带吊牌的。" **绝对禁止**主动提及任何降价信息。
    -   **如果用户是"议价"**：
        -   **分析用户出价**：
            -   如果用户出价【高于或等于】我们的【最低价】，你可以爽快地接受交易。例如：“好的，92元就出给你吧，我改下价格你拍吧。”
            -   如果用户出价【低于】我们的【最低价】，你需要礼貌地拒绝并可以尝试引导。例如：“亲，这个价格真的不行呢，最多给您优惠到92元。”
            -   如果用户只是模糊地问“能便宜点吗”，你可以给出小幅优惠，但不要直接报出最低价。可以报一个介于原价和最低价之间的价格，为后续议价留出空间。
        -   **坚守底线**：【最低价】是你的底线，**绝对不能**接受低于此价格的交易。
        -   **语气**：保持礼貌、友好的沟通风格。

【商品信息】
名称: {item_title}
原价: {item_price} 元
描述: {item_desc}

【你的议价底线】
最低价: {floor_price} 元

【当前用户消息】
{user_msg}

请严格遵守你的角色和规则，生成回复。
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
            item_title = self.item_info.get('title', '')
            item_price_str = self.item_info.get('soldPrice', '0')
            item_desc = self.item_info.get('desc', '')
            
            # 转换价格为浮点数
            try:
                item_price = float(item_price_str)
            except (ValueError, TypeError):
                item_price = 0.0

            # 计算议价底线（例如，原价的92%）
            floor_price = round(item_price * 0.92, 2)
            
            prompt = self.price_prompt.format(
                item_title=item_title,
                item_price=item_price,
                item_desc=item_desc,
                floor_price=floor_price,
                user_msg=user_msg
            )
            
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg}
            ]
            
            response = self.openai_client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "gpt-4o"),
                messages=messages,
                temperature=0.1,  # 降低随机性，让其更严格遵守指令
                max_tokens=100
            )
            
            reply = response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"PriceAgent API调用失败: {str(e)}")
            reply = "抱歉，暂时无法查询价格，请稍后再试。"

        return reply 