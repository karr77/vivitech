#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从闲鱼获取已发布商品数据并导入到Milvus向量数据库
"""

import os
import sys
import json
import logging
import argparse
from dotenv import load_dotenv

from utils.xianyu_utils import trans_cookies
from XianyuApis import XianyuApis
from db import MilvusKnowledgeBase

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main(force_update=True, skip_sold=True, keep_existing=False):
    """从闲鱼获取已发布商品数据并导入到Milvus向量数据库
    
    Args:
        force_update: 是否强制更新已存在的商品信息
        skip_sold: 是否跳过已售出商品
        keep_existing: 是否保留数据库中已存在但当前没有在闲鱼上架的商品
    
    Returns:
        int: 退出码，0表示成功，1表示失败
    """
    # 加载环境变量
    load_dotenv()
    
    # 获取cookies
    cookies_str = os.getenv('COOKIES_STR')
    if not cookies_str:
        print("错误: 环境变量COOKIES_STR未设置，请在.env文件中设置")
        return 1
    
    # 初始化Milvus知识库

    
    # 转换cookies
    cookies = trans_cookies(cookies_str)
    
    # 创建API实例
    api = XianyuApis()
    
    try:
        # 获取已发布商品列表
        print("正在获取已发布商品列表...")
        
        # 合并多页结果
        all_cards = []
        page = 1
        page_size = 20
        has_more = True
        
        while has_more:
            print(f"获取第{page}页商品列表...")
            result = api.get_published_items(cookies, page_number=page, page_size=page_size)
            
            if not (result.get('ret') and result['ret'][0].startswith('SUCCESS::')):
                print(f"错误: 获取商品列表失败: {result.get('ret')}")
                break
            
            data = result.get('data', {})
            card_list = data.get('cardList', [])
            
            if not card_list:
                has_more = False
            else:
                all_cards.extend(card_list)
                # 检查是否还有更多页
                if len(card_list) < page_size:
                    has_more = False
                page += 1
        
        if not all_cards:
            print("未找到任何已发布商品")
            return 0
            
        print(f"找到 {len(all_cards)} 个已发布商品，准备筛选和导入...")
        
        # 格式化数据为知识库格式
        knowledge_items = []
        in_stock_count = 0
        sold_out_count = 0
        
        # 获取所有在售商品ID，用于后续清理已售出商品
        in_stock_item_ids = set()
        
        for card in all_cards:
            card_data = card.get('cardData', {})
            item_id = card_data.get('id')
            if not item_id:
                continue
                
            title = card_data.get('title', '')
            price = card_data.get('priceInfo', {}).get('price', '')
            is_in_stock = card_data.get('itemStatus') == 0
            status = "在售" if is_in_stock else "已售出"
            image_url = card_data.get('picInfo', {}).get('picUrl', '') if card_data.get('picInfo') else ''
            
            # 获取商品描述
            description = card_data.get('desc', '')
            
            # 打印商品信息
            print(f"\n--- 商品 {len(knowledge_items) + sold_out_count + 1} ---")
            print(f"商品ID: {item_id}")
            print(f"标题: {title}")
            print(f"价格: {price}")
            print(f"状态: {status}")
            print(f"图片: {image_url}")
           
            # 构建基本信息
            content = f"{title}，价格: {price}元，状态: {status}"
            if description:
                content += f"，描述: {description}"
                
       
            # 获取商品详细信息
        
        
        print(f"\n筛选结果: 在售商品 {in_stock_count} 个，已售出商品 {sold_out_count} 个")
        

        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从闲鱼获取已发布商品数据并导入到Milvus向量数据库")
    parser.add_argument("--no-update", action="store_false", dest="force_update", 
                      help="不更新已存在的商品信息")
    parser.add_argument("--include-sold", action="store_false", dest="skip_sold",
                      help="包含已售出商品")
    parser.add_argument("--keep-existing", action="store_true",
                      help="保留数据库中已存在的商品")
    
    args = parser.parse_args()
    
    sys.exit(main(force_update=args.force_update, 
                 skip_sold=args.skip_sold,
                 keep_existing=args.keep_existing)) 