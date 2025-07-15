#!/bin/bash

# 清空之前的测试日志文件
> debug.log
> logs/debug_$(date +%Y-%m-%d).log

# 激活虚拟环境并执行Python脚本
source venv/bin/activate
export API_KEY="你的API密钥"
export MODEL_BASE_URL="你的中转站URL"
export MODEL_NAME="gpt-3.5-turbo"
python main.py
