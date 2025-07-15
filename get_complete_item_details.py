#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
完整获取闲鱼商品信息，包括商品列表和详细描述
"""

import os
import json
import time
import requests
import hashlib
from typing import List, Dict
from dotenv import load_dotenv

from utils.xianyu_utils import trans_cookies, generate_device_id, generate_sign

class XianyuFullDetails:
    def __init__(self):
        # 加载环境变量
        load_dotenv()
        
        # 获取cookies
        cookies_str = os.getenv('COOKIES_STR')
        if not cookies_str:
            cookies_path = os.getenv('COOKIES_PATH')
            if cookies_path and os.path.exists(cookies_path):
                with open(cookies_path, 'r') as f:
                    cookies_str = f.read().strip()
        
        if not cookies_str:
            raise ValueError("错误: 未找到cookies，请在.env文件中设置COOKIES_STR或COOKIES_PATH")
        
        # 转换cookies
        self.cookies = trans_cookies(cookies_str)
        self.user_id = self.cookies.get('unb', '')
        
        # 设置请求头
        self.headers = {
            'accept': 'application/json',
            'accept-language': 'zh-CN,zh;q=0.9',
            'content-type': 'application/x-www-form-urlencoded',
            'origin': 'https://www.goofish.com',
            'priority': 'u=1, i',
            'referer': 'https://www.goofish.com/',
            'sec-ch-ua': '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
        }
        
        # 创建会话
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.cookies.update(self.cookies)
        
    def get_item_list(self, page_size=20, max_pages=100) -> List[Dict]:
        """
        获取已发布商品列表
        
        Args:
            page_size: 每页条数
            max_pages: 最大页数
            
        Returns:
            List[Dict]: 商品列表数据
        """
        print(f"正在获取已发布商品列表...")
        
        all_items = []
        page = 1
        has_more = True
        
        while has_more and page <= max_pages:
            print(f"获取第{page}页商品列表...")
            
            # 设置参数
            timestamp = str(int(time.time() * 1000))
            device_id = generate_device_id(self.user_id)
            
            params = {
                'jsv': '2.7.2',
                'appKey': '34839810',
                't': timestamp,
                'sign': '',
                'v': '1.0',
                'type': 'originaljson',
                'accountSite': 'xianyu',
                'dataType': 'json',
                'timeout': '20000',
                'api': 'mtop.idle.web.xyh.item.list',
                'sessionOption': 'AutoLoginOnly',
                'spm_cnt': 'a21ybx.personal.0.0',
            }
            
            data_val = json.dumps({
                "needGroupInfo": True,
                "pageNumber": page,
                "userId": self.user_id,
                "pageSize": page_size
            })
            
            data = {
                'data': data_val,
            }
            
            token = self.cookies.get('_m_h5_tk', '').split('_')[0]
            sign = generate_sign(params['t'], token, data_val)
            params['sign'] = sign
            
            try:
                response = self.session.post(
                    'https://h5api.m.goofish.com/h5/mtop.idle.web.xyh.item.list/1.0/',
                    params=params,
                    data=data
                )
                
                result = response.json()
                
                if not (result.get('ret') and result['ret'][0].startswith('SUCCESS::')):
                    print(f"获取第{page}页商品失败: {result.get('ret')}")
                    break
                
                item_list = result.get('data', {}).get('cardList', [])
                
                if not item_list:
                    has_more = False
                    print(f"第{page}页没有商品，获取完成")
                else:
                    print(f"第{page}页获取到{len(item_list)}个商品")
                    all_items.extend(item_list)
                    
                    # 如果返回的商品数量小于页面大小，说明已经是最后一页
                    if len(item_list) < page_size:
                        has_more = False
                    page += 1
                    
            except Exception as e:
                print(f"获取商品列表出错: {e}")
                break
        
        print(f"总共获取到{len(all_items)}个商品")
        return all_items
    
    def get_item_detail(self, item_id: str) -> Dict:
        """
        获取单个商品的详细信息
        
        Args:
            item_id: 商品ID
            
        Returns:
            Dict: 商品详细信息
        """
        try:
            # 设置参数
            timestamp = str(int(time.time() * 1000))
            
            params = {
                'jsv': '2.7.2',
                'appKey': '34839810',
                't': timestamp,
                'sign': '',
                'v': '1.0',
                'type': 'originaljson',
                'accountSite': 'xianyu',
                'dataType': 'json',
                'timeout': '20000',
                'api': 'mtop.taobao.idle.pc.detail',
                'sessionOption': 'AutoLoginOnly',
                'spm_cnt': 'a21ybx.item.0.0',
            }
            
            data_val = f'{{"itemId":"{item_id}"}}'
            data = {
                'data': data_val,
            }
            
            token = self.cookies.get('_m_h5_tk', '').split('_')[0]
            sign = generate_sign(params['t'], token, data_val)
            params['sign'] = sign
            
            response = self.session.post(
                'https://h5api.m.goofish.com/h5/mtop.taobao.idle.pc.detail/1.0/',
                params=params,
                data=data
            )
            
            detail = response.json()
            return detail
        
        except Exception as e:
            print(f"获取商品详情出错: {e}")
            return {}
        
    def extract_full_description(self, item_detail: Dict) -> str:
        """从商品详情中提取完整描述信息"""
        try:
            # 尝试从shareData.shareInfoJsonString中提取
            share_info = item_detail.get('data', {}).get('itemDO', {}).get('shareData', {}).get('shareInfoJsonString', '')
            if share_info:
                try:
                    share_info_json = json.loads(share_info)
                    content = share_info_json.get('contentParams', {}).get('mainParams', {}).get('content', '')
                    if content:
                        return content
                except:
                    pass
            
            # 如果上面的方法失败，尝试从desc字段获取
            desc = item_detail.get('data', {}).get('itemDO', {}).get('desc', '')
            if desc:
                return desc
                
            return "无法获取完整描述"
        except Exception as e:
            print(f"提取描述信息出错: {e}")
            return "提取描述信息出错"
    
    def extract_item_info(self, card_data: Dict) -> Dict:
        """从商品列表项提取基本信息"""
        try:
            item_id = card_data.get('id', '')
            title = card_data.get('title', '')
            price = card_data.get('priceInfo', {}).get('price', '0')
            is_in_stock = card_data.get('itemStatus', -1) == 0
            status = "在售" if is_in_stock else "已售出"
            image_url = card_data.get('picInfo', {}).get('picUrl', '') if card_data.get('picInfo') else ''
            
            return {
                "id": item_id,
                "title": title,
                "price": price,
                "status": status,
                "image_url": image_url
            }
        except Exception as e:
            print(f"提取商品信息出错: {e}")
            return {}
    
    def process_items(self, limit=50):
        """处理商品列表，获取详情并输出完整信息"""
        # 获取商品列表
        item_list = self.get_item_list()
        
        # 限制处理数量
        if limit and limit < len(item_list):
            item_list = item_list[:limit]
            print(f"限制处理前{limit}个商品")
        
        results = []
        
        # 处理每个商品
        for i, item in enumerate(item_list):
            card_data = item.get('cardData', {})
            item_info = self.extract_item_info(card_data)
            
            if not item_info.get('id'):
                continue
                
            print(f"\n===== 商品 {i+1} =====")
            print(f"商品ID: {item_info['id']}")
            print(f"标题: {item_info['title']}")
            print(f"价格: {item_info['price']}")
            print(f"状态: {item_info['status']}")
            print(f"图片: {item_info['image_url']}")
            
            # 获取详情
            print(f"获取商品详情...")
            item_detail = self.get_item_detail(item_info['id'])
            
            # 提取完整描述
            full_description = self.extract_full_description(item_detail)
            print(f"\n完整描述:")
            print(f"{full_description}")
            
            # 添加到结果
            item_info['full_description'] = full_description
            results.append(item_info)
            
            # 避免请求过快
            time.sleep(1)
        
        return results
            

def main():
    try:
        # 创建实例
        xianyu = XianyuFullDetails()
        
        # 处理商品列表
        items = xianyu.process_items(limit=50)  # 处理前50个商品
        
        # 保存结果到文件
        with open('full_item_detailes.json', 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        
        print(f"\n处理完成，结果已保存到 full_item_detailes.json")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 