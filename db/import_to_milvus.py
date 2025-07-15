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
    try:
        # 从环境变量获取Milvus连接参数
        milvus_uri = os.getenv('MILVUS_URI')
        milvus_token = os.getenv('MILVUS_TOKEN')
        milvus_db = os.getenv('MILVUS_DB_NAME', 'xianyu_items')
        
        # 初始化知识库
        knowledge_base = MilvusKnowledgeBase(
            milvus_uri=milvus_uri,
            milvus_token=milvus_token,
            milvus_db=milvus_db
        )
        print("向量知识库初始化成功，已连接到Milvus服务器")
    except Exception as e:
        print(f"错误: 向量知识库初始化失败: {e}")
        return 1
    
    # 转换cookies
    cookies = trans_cookies(cookies_str)
    
    # 创建API实例
    api = XianyuApis()
    
    try:
        # 获取已发布商品列表
        print("正在获取已发布商品列表...")
        result = api.get_published_items(cookies)
        
        if not (result.get('ret') and result['ret'][0].startswith('SUCCESS::')):
            print(f"错误: 获取商品列表失败: {result.get('ret')}")
            return 1
        
        data = result.get('data', {})
        card_list = data.get('cardList', [])
        
        if not card_list:
            print("未找到任何已发布商品")
            return 0
            
        print(f"找到 {len(card_list)} 个已发布商品，准备筛选和导入...")
        
        # 格式化数据为知识库格式
        knowledge_items = []
        in_stock_count = 0
        sold_out_count = 0
        
        # 获取所有在售商品ID，用于后续清理已售出商品
        in_stock_item_ids = set()
        
        for card in card_list:
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
            
            # 只导入在售商品
            if not is_in_stock and skip_sold:
                print("状态为已售出，跳过导入")
                sold_out_count += 1
                continue
                
            if is_in_stock:
                in_stock_count += 1
                in_stock_item_ids.add(item_id)
                
            # 构建基本信息
            content = f"{title}，价格: {price}元，状态: {status}"
            if description:
                content += f"，描述: {description}"
                
            knowledge_item = {
                "itemId": item_id,
                "content": content,
                "category": "basic_info",
                "title": title,
                "price": float(price) if price and str(price).replace('.', '').isdigit() else 0.0,
                "status": status,
                "imageUrl": image_url
            }
            
            knowledge_items.append(knowledge_item)
            
            # 获取商品详细信息
            try:
                item_detail = api.get_item_info(cookies, item_id)
                if item_detail.get('ret') and item_detail['ret'][0].startswith('SUCCESS::'):
                    item_info = item_detail.get('data', {}).get('itemDO', {})
                    
                    # 提取技术规格信息
                    detail_params = item_info.get('detailParams', {})
                    if isinstance(detail_params, dict) and detail_params:
                        detail_content = []
                        
                        # 图片宽高
                        if 'picWidth' in detail_params and 'picHeight' in detail_params:
                            detail_content.append(f"图片尺寸: {detail_params.get('picWidth')}x{detail_params.get('picHeight')}")
                        
                        # 售价
                        if 'soldPrice' in detail_params:
                            detail_content.append(f"售价: {detail_params.get('soldPrice')}元")
                        
                        # 其他技术参数
                        for key, value in detail_params.items():
                            if key not in ['picWidth', 'picHeight', 'soldPrice'] and isinstance(value, (str, int, float)):
                                detail_content.append(f"{key}: {value}")
                        
                        # 由于我们不再使用属性集合，这部分技术规格信息将被忽略
            except Exception as e:
                print(f"获取商品 {item_id} 详细信息失败: {e}")
        
        print(f"\n筛选结果: 在售商品 {in_stock_count} 个，已售出商品 {sold_out_count} 个")
        
        # 导入到Milvus向量数据库
        if knowledge_items:
            print(f"准备导入 {len(knowledge_items)} 条知识项到Milvus向量数据库...")
            
            # 使用多集合模式导入
            import_results = knowledge_base.import_multi_collection(knowledge_items, force_update=force_update)
            print("多集合导入结果:")
            print(f"- 基本信息集合: {import_results['items']} 条")
            print(f"- 图片集合: {import_results['images']} 条")
            total_imported = sum(import_results.values())
            print(f"总计导入: {total_imported} 条")
            
            # 如果不保留现有商品，则清理数据库中不在当前列表中的商品
            if not keep_existing:
                # 查找数据库中不在当前在售列表中的商品（已售出或删除的商品）
                try:
                    print("正在查找已售出或已删除的商品...")
                    
                    # 构建需要删除的商品ID列表
                    # 这些是数据库中存在但不在当前在售商品列表中的商品
                    sold_item_ids = []
                    
                    # 这里可以使用条件查询从数据库中获取所有商品ID，然后与in_stock_item_ids比较
                    # 但由于我们的设计中没有直接通过条件检索所有ID的方法，我们可以从card_list中找出已售出的商品
                    for card in card_list:
                        card_data = card.get('cardData', {})
                        item_id = card_data.get('id')
                        if item_id and card_data.get('itemStatus') != 0:
                            sold_item_ids.append(item_id)
                    
                    if sold_item_ids:
                        print(f"正在从Milvus数据库中清理 {len(sold_item_ids)} 个已售出或已删除商品...")
                        # 多集合模式删除
                        remove_results = knowledge_base.remove_items(sold_item_ids)
                        print("多集合清理结果:")
                        print(f"- 基本信息集合: {remove_results['items']} 条")
                        print(f"- 图片集合: {remove_results['images']} 条")
                        total_removed = sum(remove_results.values())
                        print(f"总计清理: {total_removed} 条")
                except Exception as e:
                    print(f"清理已售出商品时出错: {e}")
        else:
            print("没有找到可导入的知识项")
        
        return 0
        
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