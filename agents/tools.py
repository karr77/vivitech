from typing import Callable, Any

class Tool:
    """
    一个标准化的工具类，用于包装Agent的功能。
    借鉴了LangChain等框架的设计思想，为未来的架构扩展做准备。
    """
    def __init__(self, name: str, description: str, func: Callable):
        """
        初始化工具。

        Args:
            name (str): 工具的唯一名称，用于LLM的识别和调用。
            description (str): 对工具功能的详细描述，将用于生成Prompt，帮助LLM理解何时使用该工具。
            func (Callable): 该工具要执行的实际函数或方法。
        """
        self.name = name
        self.description = description
        self.func = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """使工具实例可调用。"""
        return self.func(*args, **kwargs)

    def __str__(self) -> str:
        return f"Tool(name='{self.name}', description='{self.description}')" 