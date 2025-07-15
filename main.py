import base64
import json
import asyncio
import time
import os
import websockets
from loguru import logger
from dotenv import load_dotenv
from XianyuApis import XianyuApis
import sys
import requests
from openai import OpenAI


from utils.xianyu_utils import generate_mid, generate_uuid, trans_cookies, generate_device_id, decrypt
from context_manager import ChatContextManager

# 导入协调器
from agents.coordinator import CoordinatorAgent
from hybrid_retriever import XianyuHybridRetrieval

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{message}</level>")

class XianyuLive:
    def __init__(self, cookies_str=None):
        """初始化XianyuLive类，设置所有初始配置和变量。"""
        self.xianyu = XianyuApis()
        self.base_url = 'wss://wss-goofish.dingtalk.com/'
        self.cookies_str = cookies_str
        
        if cookies_str is None:
            self.cookies = {}
            self.myid = os.getenv("DEFAULT_USER_ID", "0000000000")
            self.device_id = generate_device_id(self.myid)
        else:
            self.cookies = trans_cookies(cookies_str)
            self.xianyu.session.cookies.update(self.cookies)
            self.myid = self.cookies.get('unb', os.getenv("DEFAULT_USER_ID", "0000000000"))
            self.device_id = generate_device_id(self.myid)
            
        self.context_manager = ChatContextManager()
        
        self.heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL", "15"))
        self.heartbeat_timeout = int(os.getenv("HEARTBEAT_TIMEOUT", "5"))
        self.last_heartbeat_response = 0
        self.ws = None
        
        self.token_refresh_interval = int(os.getenv("TOKEN_REFRESH_INTERVAL", "3600"))
        self.token_retry_interval = int(os.getenv("TOKEN_RETRY_INTERVAL", "300"))
        self.last_token_refresh_time = 0
        self.current_token = None
        self.connection_restart_flag = False
        
        # 人工客服智能避让相关配置
        self.human_reply_silence_period = int(os.getenv("HUMAN_REPLY_SILENCE_PERIOD", "300")) # 人工回复后，机器人保持沉默的时长（秒）
        self.human_last_reply_timestamps = {} # 记录每个人工会话的最后回复时间

        self.message_expire_time = int(os.getenv("MESSAGE_EXPIRE_TIME", "300000"))
        self.toggle_keywords = os.getenv("TOGGLE_KEYWORDS", "。")
        
        self._init_multi_agent_system()

    def _init_multi_agent_system(self):
        """初始化多智能体系统，包括加载数据和创建协调器Agent。"""
        logger.info("Initializing multi-agent system...")
        
        # 1. 初始化混合检索器 (如果需要)
        self.hybrid_retriever = XianyuHybridRetrieval()
        use_mock_data = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
        if use_mock_data:
            try:
                with open('data/cleaned_product.json', 'r', encoding='utf-8') as f:
                    products_data = json.load(f)
                self.hybrid_retriever.load_products(products_data)
                logger.info("Mock product data loaded for Hybrid Retriever.")
            except Exception as e:
                logger.warning(f"Could not load mock data: {e}")
        
        # 2. 初始化OpenAI客户端
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("MODEL_BASE_URL", "https://api.openai.com/v1")
        openai_client = OpenAI(api_key=api_key, base_url=base_url)

        # 3. 初始化总协调器
        # CoordinatorAgent现在内部管理所有专家Agent
        self.coordinator = CoordinatorAgent(
            openai_client=openai_client, 
            hybrid_retriever=self.hybrid_retriever
        )
        logger.info("Multi-agent system initialized successfully with LlamaIndex-driven Coordinator.")

    async def refresh_token(self):
        """调用XianyuApis刷新WebSocket连接所需的token。"""
        logger.info("正在尝试刷新Token...")
        token_result = self.xianyu.get_token(self.cookies, self.device_id)
        if 'data' in token_result and 'accessToken' in token_result['data']:
            self.current_token = token_result['data']['accessToken']
            self.last_token_refresh_time = time.time()
            logger.info(f"Token刷新成功: {self.current_token[:10]}...")
            return self.current_token
        
        logger.error(f"Token获取失败，API返回: {token_result}")
        return None

    async def token_refresh_loop(self):
        """定期检查并刷新token的后台循环任务。"""
        while True:
            if time.time() - self.last_token_refresh_time >= self.token_refresh_interval:
                new_token = await self.refresh_token()
                if new_token:
                    self.connection_restart_flag = True
                    if self.ws:
                        await self.ws.close()
                    break
                else:
                    await asyncio.sleep(self.token_retry_interval)
            await asyncio.sleep(60)

    async def send_msg(self, ws, cid, toid, text):
        """向指定用户发送格式化的闲鱼聊天消息。"""
        text_content = {"contentType": 1, "text": {"text": text}}
        text_base64 = str(base64.b64encode(json.dumps(text_content).encode('utf-8')), 'utf-8')
        msg = {
            "lwp": "/r/MessageSend/sendByReceiverScope",
            "headers": {"mid": generate_mid()},
            "body": [
                {
                    "uuid": generate_uuid(), "cid": f"{cid}@goofish", "conversationType": 1,
                    "content": {"contentType": 101, "custom": {"type": 1, "data": text_base64}},
                    "redPointPolicy": 0, "extension": {"extJson": "{}"},
                    "ctx": {"appVersion": "1.0", "platform": "web"},
                    "mtags": {}, "msgReadStatusSetting": 1
                },
                {"actualReceivers": [f"{toid}@goofish", f"{self.myid}@goofish"]}
            ]
        }
        await ws.send(json.dumps(msg))

    async def init(self, ws):
        """初始化WebSocket连接，包括注册和同步状态。"""
        if not self.current_token:
            await self.refresh_token()
        
        if not self.current_token:
            raise Exception("Token获取失败")
            
        msg_reg = {
            "lwp": "/reg",
            "headers": {
                "cache-header": "app-key token ua wv",
                "app-key": "444e9908a51d1cb236a27862abc769c9",
                "token": self.current_token,
                "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 DingTalk(2.1.5) OS(Windows/10) Browser(Chrome/133.0.0.0) DingWeb/2.1.5 IMPaaS DingWeb/2.1.5",
                "dt": "j", "wv": "im:3,au:3,sy:6", "sync": "0,0;0;0;",
                "did": self.device_id, "mid": generate_mid()
            }
        }
        await ws.send(json.dumps(msg_reg))
        await asyncio.sleep(1)
        msg_ack = {"lwp": "/r/SyncStatus/ackDiff", "headers": {"mid": "5701741704675979 0"}, "body": [
            {"pipeline": "sync", "tooLong2Tag": "PNM,1", "channel": "sync", "topic": "sync", "highPts": 0,
             "pts": int(time.time() * 1000) * 1000, "seq": 0, "timestamp": int(time.time() * 1000)}]}
        await ws.send(json.dumps(msg_ack))

    def is_chat_message(self, message):
        """判断接收到的消息是否为用户发送的聊天消息。"""
        return (
            isinstance(message, dict) and "1" in message and isinstance(message.get("1"), dict) and
            "10" in message["1"] and isinstance(message["1"].get("10"), dict) and
            "reminderContent" in message["1"]["10"]
        )

    async def handle_message(self, message_data, websocket):
        """处理接收到的各类WebSocket消息的核心函数。"""
        try:
            sync_data = message_data["body"]["syncPushPackage"]["data"][0]
            if "data" not in sync_data: return

            try:
                json.loads(base64.b64decode(sync_data["data"]).decode("utf-8"))
                return
            except Exception:
                message = json.loads(decrypt(sync_data["data"]))
        except (KeyError, IndexError, TypeError):
            return

        if not self.is_chat_message(message):
            return

        create_time = int(message["1"]["5"])
        if (time.time() * 1000 - create_time) > self.message_expire_time:
            return

        send_user_id = message["1"]["10"]["senderUserId"]
        send_message = message["1"]["10"]["reminderContent"]
        url_info = message["1"]["10"]["reminderUrl"]
        item_id = url_info.split("itemId=")[1].split("&")[0] if "itemId=" in url_info else None
        chat_id = message["1"]["2"].split('@')[0]
        
        if not item_id: return

        # 如果是人工客服（自己）发的消息
        if send_user_id == self.myid:
            # 记录下人工客服的最后回复时间，并启动静默期
            self.human_last_reply_timestamps[chat_id] = time.time()
            logger.info(f"检测到人工客服回复，会话 {chat_id} 进入 {self.human_reply_silence_period}秒 静默期。")
            return
        
        # 如果是用户发来的消息
        logger.info(f"◀️ RECV from {send_user_id}: {send_message}")
        self.context_manager.add_message_by_chat(chat_id, send_user_id, item_id, "user", send_message)
        
        # 在机器人回复前，检查人工客服是否在静默期内
        if chat_id in self.human_last_reply_timestamps and \
           (time.time() - self.human_last_reply_timestamps[chat_id]) < self.human_reply_silence_period:
            logger.info(f"人工客服静默期内 (会话: {chat_id})，机器人本次不响应。")
            return

        item_info = self.context_manager.get_item_info(item_id)
        if not item_info:
            api_result = self.xianyu.get_item_info(item_id=item_id)
            if 'data' in api_result and 'itemDO' in api_result['data']:
                item_info = api_result['data']['itemDO']
                self.context_manager.save_item_info(item_id, item_info)
            else:
                return
        
        if not hasattr(self, 'coordinator'):
            return

        # 使用新的LlamaIndex Agent处理流程
        logger.info(f"Handing off to LlamaIndex agent for session: {chat_id}")
        
        # 1. 为当前会话和商品获取一个配置好的Agent Runner
        agent_runner = self.coordinator.get_agent_runner(session_id=chat_id, item_info=item_info)
        
        # 2. 使用astream_chat获取异步流式响应
        response_stream = await agent_runner.astream_chat(send_message)
        
        # 3. 从流中拼接最终回复
        bot_reply_chunks = []
        async for chunk in response_stream.async_response_gen():
            bot_reply_chunks.append(chunk)
        bot_reply = "".join(bot_reply_chunks).strip()

        if not bot_reply:
            logger.warning("Agent returned an empty reply, skipping.")
            return

        self.context_manager.add_message_by_chat(chat_id, self.myid, item_id, "assistant", bot_reply)
        logger.info(f"▶️ SEND to {send_user_id}: {bot_reply}")
        await self.send_msg(websocket, chat_id, send_user_id, bot_reply)

    async def heartbeat_loop(self, ws):
        """维持WebSocket连接的心跳循环任务。"""
        last_heartbeat_time = time.time()
        while True:
            if time.time() - last_heartbeat_time >= self.heartbeat_interval:
                await ws.send(json.dumps({"lwp": "/!", "headers": {"mid": generate_mid()}}))
                last_heartbeat_time = time.time()
            
            if (time.time() - self.last_heartbeat_response) > (self.heartbeat_interval + self.heartbeat_timeout):
                break
            await asyncio.sleep(1)

    async def main(self):
        """程序主入口，负责建立和维护WebSocket连接，并处理消息循环。"""
        while True:
            self.connection_restart_flag = False
            headers = {
                "Cookie": self.cookies_str, "Host": "wss-goofish.dingtalk.com", "Connection": "Upgrade",
                "Pragma": "no-cache", "Cache-Control": "no-cache",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
                "Origin": "https://www.goofish.com", "Accept-Encoding": "gzip, deflate, br, zstd",
                "Accept-Language": "zh-CN,zh;q=0.9",
            }
            try:
                async with websockets.connect(self.base_url, extra_headers=headers) as websocket:
                    self.ws = websocket
                    await self.init(websocket)
                    
                    self.last_heartbeat_response = time.time()
                    
                    heartbeat_task = asyncio.create_task(self.heartbeat_loop(websocket))
                    token_refresh_task = asyncio.create_task(self.token_refresh_loop())
                    
                    async for message in websocket:
                        if self.connection_restart_flag: break
                        message_data = json.loads(message)
                        
                        if message_data.get("code") == 200 and "mid" in message_data.get("headers", {}):
                            self.last_heartbeat_response = time.time()
                            continue
                        
                        if "mid" in message_data.get("headers", {}):
                            ack = {"code": 200, "headers": {"mid": message_data["headers"]["mid"]}}
                            await websocket.send(json.dumps(ack))
                        
                        if "syncPushPackage" in message_data.get("body", {}):
                            await self.handle_message(message_data, websocket)
            
            except (websockets.exceptions.ConnectionClosed, Exception) as e:
                logger.error(f"主循环中发生致命错误: {e}")
                import traceback
                traceback.print_exc()
                
            finally:
                if 'heartbeat_task' in locals() and not heartbeat_task.done():
                    heartbeat_task.cancel()
                if 'token_refresh_task' in locals() and not token_refresh_task.done():
                    token_refresh_task.cancel()
                
                if not self.connection_restart_flag:
                    await asyncio.sleep(5)

if __name__ == '__main__':
    load_dotenv()
    cookies_str = os.getenv("COOKIES_STR")
    if not cookies_str:
        logger.error("未在.env文件中找到COOKIES_STR，程序退出。")
        sys.exit(1)
        
    xianyuLive = XianyuLive(cookies_str)
    asyncio.run(xianyuLive.main())
