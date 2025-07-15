import json
import time

import requests
from loguru import logger

from utils.xianyu_utils import generate_sign, trans_cookies, generate_device_id


class XianyuApis:
    def __init__(self):
        self.url = 'https://h5api.m.goofish.com/h5/mtop.taobao.idlemessage.pc.login.token/1.0/'
        self.session = requests.Session()
        self.headers = {
            'accept': 'application/json',
            'accept-language': 'zh-CN,zh;q=0.9',
            'cache-control': 'no-cache',
            'origin': 'https://www.goofish.com',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://www.goofish.com/',
            'sec-ch-ua': '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
        }
        
        # 设置session headers
        self.session.headers.update(self.headers)
        
    def get_token(self, cookies, device_id):

        params = {
            'jsv': '2.7.2',
            'appKey': '34839810',
            't': str(int(time.time()) * 1000),
            'sign': '',
            'v': '1.0',
            'type': 'originaljson',
            'accountSite': 'xianyu',
            'dataType': 'json',
            'timeout': '20000',
            'api': 'mtop.taobao.idlemessage.pc.login.token',
            'sessionOption': 'AutoLoginOnly',
            'spm_cnt': 'a21ybx.im.0.0',
        }
        data_val = '{"appKey":"444e9908a51d1cb236a27862abc769c9","deviceId":"' + device_id + '"}'
        data = {
            'data': data_val,
        }
        token = cookies['_m_h5_tk'].split('_')[0]
        sign = generate_sign(params['t'], token, data_val)
        params['sign'] = sign
        response = self.session.post('https://h5api.m.goofish.com/h5/mtop.taobao.idlemessage.pc.login.token/1.0/', params=params, cookies=cookies, data=data)
        res_json = response.json()
        return res_json

    def get_item_info(self, item_id):
        """
        获取商品信息
        
        Args:
            item_id: 商品ID
            
        Returns:
            dict: 商品信息
        """
        # 使用session中已有的cookies
        cookies = self.session.cookies
        
        # 安全检查：确保cookies存在并包含必要的令牌
        if not cookies or '_m_h5_tk' not in cookies:
            logger.error("获取商品信息失败：cookies不存在或缺少必要的令牌")
            return {
                "ret": ["FAIL::cookies不存在或缺少必要的令牌"],
                "data": {
                    "mock": True,
                    "desc": "演示商品 - API调用失败",
                    "soldPrice": "9999.00"
                }
            }

        params = {
            'jsv': '2.7.2',
            'appKey': '34839810',
            't': str(int(time.time()) * 1000),
            'sign': '',
            'v': '1.0',
            'type': 'originaljson',
            'accountSite': 'xianyu',
            'dataType': 'json',
            'timeout': '20000',
            'api': 'mtop.taobao.idle.pc.detail',
            'sessionOption': 'AutoLoginOnly',
            'spm_cnt': 'a21ybx.im.0.0',
        }
        data_val = '{"itemId":"' + item_id + '"}'
        data = {
            'data': data_val,
        }
        
        try:
            token = cookies['_m_h5_tk'].split('_')[0]
            sign = generate_sign(params['t'], token, data_val)
            params['sign'] = sign
            response = self.session.post('https://h5api.m.goofish.com/h5/mtop.taobao.idle.pc.detail/1.0/', params=params, data=data)
            res_json = response.json()
            return res_json
        except Exception as e:
            logger.error(f"获取商品信息失败：{e}")
            return {
                "ret": [f"FAIL::{str(e)}"],
                "data": {
                    "mock": True,
                    "desc": "演示商品 - 商品加载失败",
                    "soldPrice": "9999.00"
                }
            }

    def get_published_items(self, cookies, user_id=None, page_number=1, page_size=20):
        """
        获取已发布的商品列表
        
        Args:
            cookies: 用户cookies
            user_id: 用户ID，默认从cookies中获取
            page_number: 页码，默认1
            page_size: 每页条数，默认20
            
        Returns:
            dict: 商品列表数据，结构如下:
                {
                    "ret": ["SUCCESS::接口调用成功"],
                    "data": {
                        "cardList": [
                            {
                                "cardType": 1003,
                                "cardData": {
                                    "id": "商品ID",
                                    "title": "商品标题",
                                    "priceInfo": {
                                        "price": "价格"
                                    },
                                    "itemStatus": 状态码(0=在售,1=已售出),
                                    "pic": {
                                        "url": "图片URL"
                                    }
                                }
                            }
                        ]
                    }
                }
        """
        if not user_id:
            user_id = cookies.get('unb')
            
        if not user_id:
            raise ValueError("未找到用户ID，请提供user_id参数或确保cookies中包含unb字段")

        params = {
            'jsv': '2.7.2',
            'appKey': '34839810',
            't': str(int(time.time()) * 1000),
            'sign': '',
            'v': '1.0',
            'type': 'originaljson',
            'accountSite': 'xianyu',
            'dataType': 'json',
            'timeout': '20000',
            'api': 'mtop.idle.web.xyh.item.list',
            'sessionOption': 'AutoLoginOnly',
            'spm_cnt': 'a21ybx.personal.0.0',
            'spm_pre': 'a21ybx.home.nav.1.4c053da64aeMaX',
            'log_id': '4c053da64aeMaX',
        }
        
        data_val = json.dumps({
            "needGroupInfo": True,
            "pageNumber": page_number,
            "userId": user_id,
            "pageSize": page_size
        })
        
        data = {
            'data': data_val,
        }
        
        token = cookies['_m_h5_tk'].split('_')[0]
        sign = generate_sign(params['t'], token, data_val)
        params['sign'] = sign
        
        response = self.session.post(
            'https://h5api.m.goofish.com/h5/mtop.idle.web.xyh.item.list/1.0/',
            params=params,
            cookies=cookies,
            data=data
        )
        
        res_json = response.json()
        return res_json


