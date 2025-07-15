"""
闲鱼智能客服 - Agent基类定义
定义了所有Agent的标准接口
"""

from typing import Dict, Any
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Agent基类，定义标准接口
    所有专门Agent都需要继承此类并实现process方法
    """
    
    def __init__(self, item_info: Dict[str, Any]):
        """
        初始化Agent基类
        
        Args:
            item_info (Dict[str, Any]): 当前请求的商品信息
        """
        self.item_info = item_info
    
    @abstractmethod
    async def process(self, user_msg: str) -> str:
        """
        处理用户消息的核心方法
        
        Args:
            user_msg: 用户消息
            
        Returns:
            str: 回复内容
        """
        pass
    
    def __str__(self):
        return f"{self.__class__.__name__}" 