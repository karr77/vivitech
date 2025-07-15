"""
闲鱼智能客服 - 协调器Agent
负责分析用户意图，调度合适的专业Agent，并整合各Agent回复
"""

import asyncio
import json
import os
from typing import Dict, List, Any
from loguru import logger
from openai import OpenAI

from .base import BaseAgent
from .tools import Tool

# LlamaIndex Components
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI as LlamaOpenAI

# Import all specialist agents
from .retrieval_agent import RetrievalAgent
from .price_agent import PriceAgent
from .tech_agent import TechAgent
from .order_agent import OrderAgent
from .chat_agent import ChatAgent

# A new, more powerful system prompt for the ReActAgent
SYSTEM_PROMPT = """
你是闲鱼平台上一款名叫"咸鱼拍档"的AI客服助手。

你的职责是：
1. **理解用户意图**：分析用户的每一条消息，理解他们是想咨询商品细节、议价、问订单，还是只是闲聊。
2. **选择合适的工具**：根据用户意图，从你拥有的工具箱中选择最合适的工具来处理请求。你一次只能使用一个工具。
3. **驱动工具执行**：调用选定的工具，并根据工具返回的结果生成回复。
4. **保持对话连贯**：你拥有记忆能力，能记住之前的对话内容。

你的工具箱包含以下工具：
- **retrieval_tool**: 用于回答关于商品的一般性问题，如功能、用途、特点或商品间的比较。如果问题包含具体的型号代码（如 4QUT5M8），请勿使用此工具。
- **tech_tool**: 必须优先使用此工具来检查用户询问的特定商品型号是否存在或有货。也可用于回答技术规格、尺寸、材质等硬核问题。
- **price_tool**: 仅当用户明确询问价格或试图议价（例如"多少钱"、"能便宜点吗"）时使用。
- **order_tool**: 当用户询问关于下单、发货、物流、退换货政策等与交易流程相关的问题时使用。
- **chat_tool**: 当用户的消息是打招呼、闲聊或者其他工具都无法处理的非商品相关问题时使用。这是最后的兜底工具。

工作流程：
1. 分析用户的最新消息。
2. 思考应该使用哪个工具。
3. 调用所选工具。
4. 根据工具的输出，向用户生成回复。
5. 如果工具的回复已经能回答用户问题，就直接回复。如果还需要信息，可以继续调用工具。

请始终以友好、口语化的方式与用户交流。
"""

class CoordinatorAgent:
    """
    协调器Agent (LlamaIndex ReActAgent驱动版)
    
    负责:
    1. 根据会话ID管理对话记忆
    2. 为每个请求动态创建包含商品信息的工具集
    3. 初始化并运行ReActAgent来处理用户请求
    """

    def __init__(self, openai_client: OpenAI, hybrid_retriever: Any = None):
        """
        初始化协调器Agent。
        
        Args:
            openai_client (OpenAI): OpenAI客户端实例.
            hybrid_retriever (Any, optional): 混合检索器实例. Defaults to None.
        """
        self.openai_client = openai_client
        self.hybrid_retriever = hybrid_retriever
        
        # 将传入的openai客户端包装成LlamaIndex的LLM对象
        self.llm = LlamaOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o"), 
            api_key=self.openai_client.api_key,
            api_base=str(self.openai_client.base_url)
        )
        
        # 用于存储不同会话的内存
        self.sessions: Dict[str, ChatMemoryBuffer] = {}
        logger.info("LlamaIndex驱动的协调器Agent初始化完成")

    def _get_tools(self, item_info: Dict[str, Any]) -> List[FunctionTool]:
        """为当前请求动态创建工具列表"""
        
        # 为当前请求创建专用的Agent实例（包含item_info）
        retrieval_agent = RetrievalAgent(item_info, self.hybrid_retriever, self.openai_client)
        price_agent = PriceAgent(self.openai_client, item_info)
        tech_agent = TechAgent(self.openai_client, item_info)
        order_agent = OrderAgent(self.openai_client, item_info)
        chat_agent = ChatAgent(self.openai_client, item_info)
        
        # 将Agent的方法包装成LlamaIndex的FunctionTool
        tools = [
            FunctionTool.from_defaults(
                fn=retrieval_agent.process,
                name="retrieval_tool",
                description="用于回答关于商品的一般性问题，如功能、用途、特点或商品间的比较。如果问题包含具体的型号代码（如 4QUT5M8），请勿使用此工具。"
            ),
            FunctionTool.from_defaults(
                fn=tech_agent.process,
                name="tech_tool",
                description="必须优先使用此工具来检查用户询问的特定商品型号是否存在或有货。也可用于回答技术规格、尺寸、材质等硬核问题。"
            ),
            FunctionTool.from_defaults(
                fn=price_agent.process,
                name="price_tool",
                description='仅当用户明确询问价格或试图议价（例如"多少钱"、"能便宜点吗"）时使用。'
            ),
            FunctionTool.from_defaults(
                fn=order_agent.process,
                name="order_tool",
                description='当用户询问关于下单、发货、物流、退换货政策等与交易流程相关的问题时使用。'
            ),
            FunctionTool.from_defaults(
                fn=chat_agent.process,
                name="chat_tool",
                description='当用户的消息是打招呼、闲聊或者其他工具都无法处理的非商品相关问题时使用。这是最后的兜底工具。'
            ),
        ]
        return tools

    def get_agent_runner(self, session_id: str, item_info: Dict[str, Any]) -> ReActAgent:
        """
        获取一个为特定会话和商品配置好的Agent运行器
        """
        # 1. 获取或创建当前会话的记忆
        memory = self.sessions.setdefault(
            session_id, ChatMemoryBuffer.from_defaults(token_limit=4096)
        )
        
        # 2. 为当前商品信息创建工具集
        tools = self._get_tools(item_info)

        # 3. 创建并返回ReActAgent实例
        return ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            memory=memory,
            system_prompt=SYSTEM_PROMPT,
            verbose=True  # 在后台打印Agent的思考过程，方便调试
        )

    async def generate_reply(self,
                             user_msg: str,
                             item_info: Dict[str, Any],
                             context: List[Dict],
                             selected_tool_names: List[str],
                             **kwargs) -> str:
        """
        生成回复
        
        Args:
            user_msg: 用户消息
            item_info: 商品信息 
            context: 对话上下文
            selected_tool_names: 已选定的工具名称列表
            
        Returns:
            str: 最终回复
        """
        # 1. 分析意图，确定调用哪些工具 (步骤已移至外部调用)
        logger.info(f"接收到预选工具: {selected_tool_names}")
        
        # 2. 如果没有选择任何工具，使用闲聊工具作为兜底
        if not selected_tool_names and 'chat_tool' in self.tools:
            selected_tool_names = ['chat_tool']
        
        # 3. 并行调用所有选择的工具
        results = []
        tasks = []
        
        for tool_name in selected_tool_names:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                task = asyncio.create_task(
                    tool.func(
                        user_msg=user_msg,
                        item_info=item_info,
                        context=context,
                        **kwargs
                    )
                )
                tasks.append((tool_name, task))
        
        # 等待所有任务完成
        for tool_name, task in tasks:
            try:
                result = await task
                result['agent'] = tool_name  # 标记是哪个工具的结果
                results.append(result)
                logger.debug(f"工具 {tool_name} 回复: {result['response'][:50]}...")
            except Exception as e:
                logger.error(f"工具 {tool_name} 处理失败: {e}")
        
        # 4. 整合结果生成最终回复
        if not results:
            logger.warning("没有工具返回有效结果")
            return "抱歉，无法理解您的问题，请您换个方式提问。"
        
        final_reply = self._integrate_results(results, user_msg)
        return final_reply
    
    def _integrate_results(self, results: List[Dict], user_msg: str) -> str:
        """整合多个工具的结果"""
        # 如果只有一个结果，直接返回
        if len(results) == 1:
            return results[0]['response']
        
        # 如果有多个结果，但其中一个是chat，则优先选择非chat的结果
        if len(results) > 1 and any('chat' in r['agent'] for r in results):
            non_chat_results = [r for r in results if 'chat' not in r['agent']]
            if non_chat_results:
                # 如果有其他工具的结果，则返回置信度最高的那个
                sorted_results = sorted(non_chat_results, key=lambda x: x.get('confidence', 0), reverse=True)
                return sorted_results[0]['response']
        
        # 如果剩下多个结果（或只有chat结果），按置信度排序
        sorted_results = sorted(results, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # 默认返回最高置信度的回复
        return sorted_results[0]['response'] 