import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CleanedProduct:
    """清洗后的商品数据结构"""
    id: str
    brand: str
    model: str
    product_type: str
    color: str
    size: str
    capacity: str
    price: float
    key_features: List[str]
    embedding_text: str

class ProductDataCleaner:
    def __init__(self):
        # 品牌映射 - 扩展更多变体
        self.brand_mapping = {
            '北极狐': 'Fjallraven',
            'fjallraven': 'Fjallraven', 
            '狐狸': 'Fjallraven',
            'kanken': 'Fjallraven',
            'jansport': 'JanSport',
            '杰斯伯': 'JanSport',
            '杰思博': 'JanSport',
            'js': 'JanSport',
            'dickies': 'Dickies',
            'doughnut': 'Doughnut',
            '甜甜圈': 'Doughnut',
            'lululemon': 'Lululemon',
            '露露': 'Lululemon',
            '瑜伽': 'Lululemon'
        }
        
        # 商品类型关键词 - 更全面的分类
        self.product_types = {
            '双肩包': '双肩包',
            '背包': '双肩包', 
            '书包': '双肩包',
            '电脑包': '电脑包',
            '托特包': '托特包',
            '腰包': '腰包',
            '旅行包': '旅行包',
            '信封包': '信封包',
            '单肩包': '单肩包',
            '斜挎包': '斜挎包'
        }
        
        # 颜色关键词 - 增加更多颜色变体
        self.colors = [
            '黑色', '白色', '红色', '蓝色', '绿色', '黄色', '粉色', '灰色', 
            '橙色', '紫色', '棕色', '米色', '深蓝', '海蓝', '军绿', '雾灰',
            '芥末黄', '酒红', '奶黄', '荧光绿', '蜜蜡粉', '松林绿', '深蓝色',
            '海蓝色', '军绿色', '雾灰色', '酒红色', '奶黄色', '荧光绿色'
        ]
        
        # 尺寸规格
        self.sizes = ['小号', '中号', '大号', '7L', '16L', '20L', '25L', '30L', '31L']
        
        # 价格相关词汇
        self.price_keywords = ['元', '块', '钱', '价格', '售价', '原价']

    def extract_brand(self, text: str) -> str:
        """提取品牌 - 改进匹配逻辑"""
        text_lower = text.lower()
        
        # 按品牌名长度排序，优先匹配长的品牌名避免误匹配
        sorted_brands = sorted(self.brand_mapping.items(), key=lambda x: len(x[0]), reverse=True)
        
        for key, brand in sorted_brands:
            if key.lower() in text_lower:
                return brand
        return "未知品牌"

    def extract_model(self, text: str) -> str:
        """提取型号 - 改进匹配精度"""
        # 更精确的型号匹配模式
        patterns = [
            r'4QU[ET][A-Z0-9]{3,5}',  # JanSport 4QUE/4QUT系列
            r'23\d{3}',               # 北极狐23xxx系列  
            r'T50\d{1,3}[A-Z]?',      # JanSport T50系列
            r'TYP\d{1,3}[A-Z]?',      # JanSport TYP系列
            r'JS00T\d{3}',            # JanSport JS00T系列
            r'T\d{3}[A-Z]?',          # 通用T系列
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0].upper()
                
        return ""

    def extract_color(self, text: str) -> str:
        """提取颜色 - 改进匹配逻辑"""
        # 按颜色名长度排序，优先匹配长的颜色名
        sorted_colors = sorted(self.colors, key=len, reverse=True)
        
        for color in sorted_colors:
            if color in text:
                return color
        return ""

    def extract_price_info(self, text: str) -> Tuple[str, List[int]]:
        """提取价格信息和价格数值"""
        price_pattern = r'(\d+)\s*[元块钱]'
        prices = [int(match) for match in re.findall(price_pattern, text)]
        
        # 提取价格相关描述
        price_desc = ""
        if '便宜' in text or '低价' in text or '特价' in text:
            price_desc += "便宜 "
        if '贵' in text or '高端' in text or '奢华' in text:
            price_desc += "高端 "
            
        return price_desc.strip(), prices

    def extract_size_capacity(self, text: str) -> tuple:
        """提取尺寸和容量"""
        size = ""
        capacity = ""
        
        # 按长度排序，优先匹配具体的容量
        sorted_sizes = sorted(self.sizes, key=len, reverse=True)
        
        for s in sorted_sizes:
            if s in text:
                if 'L' in s and s.replace('L', '').isdigit():
                    capacity = s
                else:
                    size = s
                break  # 找到第一个就停止
        
        return size, capacity

    def extract_key_features(self, text: str) -> List[str]:
        """提取关键特征 - 扩展特征词库"""
        features = []
        
        # 扩展的功能特征词库
        feature_keywords = [
            # 材质特征
            '防泼水', '防水', '帆布', '尼龙', '牛津布', '麂皮', '皮革', 'PU皮',
            # 功能特征  
            '防爆拉链', '反光标', '可调节', '电脑隔层', '隔层', '侧袋', '水杯袋',
            # 品质标识
            '全新', '正品', '带吊牌', '包装盒', '防伪',
            # 联名系列
            '迪士尼联名', '联名款', '限定款', '绝版',
            # 设计特点
            '大容量', '轻便', '简约', '时尚', '复古', '潮流',
            # 使用场景
            '户外', '通勤', '旅行', '学生', '商务'
        ]
        
        for keyword in feature_keywords:
            if keyword in text:
                features.append(keyword)
        
        return list(set(features))  # 去重

    def extract_product_type(self, text: str) -> str:
        """提取商品类型"""
        # 按类型名长度排序，优先匹配更具体的类型
        sorted_types = sorted(self.product_types.items(), key=lambda x: len(x[0]), reverse=True)
        
        for keyword, product_type in sorted_types:
            if keyword in text:
                return product_type
        return "背包"

    def clean_description(self, description: str) -> str:
        """清洗描述文本，移除营销内容"""
        # 移除表情符号和特殊字符
        description = re.sub(r'\[.*?\]', '', description)
        description = re.sub(r'[！!]{2,}', '！', description)
        description = re.sub(r'[…\.]{3,}', '...', description)
        
        # 分行处理
        lines = description.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # 跳过垃圾内容
            skip_patterns = [
                r'tag[：:]',
                r'#.*#',
                r'万仔商店.*',
                r'.*招待所.*',
                r'.*商店.*网上冲浪.*',
                r'.*假一[赔赔]十.*',
                r'.*支持.*验货.*',
                r'.*得物.*',
                r'.*闲鱼.*',
            ]
            
            should_skip = False
            for pattern in skip_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    should_skip = True
                    break
            
            if not should_skip and line and len(line) < 100:  # 保留有用的短句
                cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)

    def generate_embedding_text(self, product: Dict) -> str:
        """生成用于embedding的文本 - 优化语义匹配"""
        title = product.get('title', '')
        desc = product.get('full_description', '')
        full_text = title + ' ' + desc
        
        # 提取结构化信息
        brand = self.extract_brand(full_text)
        model = self.clean_model(self.extract_model(full_text))
        product_type = self.extract_product_type(full_text)
        color = self.extract_color(full_text)
        size, capacity = self.extract_size_capacity(full_text)
        features = self.extract_key_features(full_text)
        price_desc, prices = self.extract_price_info(full_text)
        cleaned_desc = self.clean_description(desc)
        
        # 构建多层次的embedding文本
        # 1. 核心属性（用于精确匹配）
        core_attrs = []
        if brand != "未知品牌":
            core_attrs.append(f"品牌：{brand}")
        if model:
            core_attrs.append(f"型号：{model}")
        core_attrs.append(f"类型：{product_type}")
        if color:
            core_attrs.append(f"颜色：{color}")
        if size:
            core_attrs.append(f"尺寸：{size}")
        if capacity:
            core_attrs.append(f"容量：{capacity}")
        core_attrs.append(f"价格：{product['price']}元")
        
        # 2. 特征描述（用于语义匹配）
        feature_text = ""
        if features:
            feature_text = f"特征：{' '.join(features)}"
        if price_desc:
            feature_text += f" {price_desc}"
            
        # 3. 语义描述（增强理解）
        semantic_parts = []
        
        # 品牌语义扩展
        brand_semantic = {
            'Fjallraven': '北极狐 瑞典 户外 kanken 经典',
            'JanSport': '杰斯伯 美国 学生 校园 经典',
            'Dickies': '工装 休闲 简约',
            'Doughnut': '甜甜圈 港风 时尚',
            'Lululemon': '瑜伽 运动 轻奢 加拿大'
        }
        if brand in brand_semantic:
            semantic_parts.append(brand_semantic[brand])
            
        # 用途语义扩展
        if '电脑' in full_text or '隔层' in full_text:
            semantic_parts.append('办公 商务 学习 电脑')
        if '学生' in full_text or '书包' in full_text:
            semantic_parts.append('上学 读书 青春 校园')
        if '旅行' in full_text or '户外' in full_text:
            semantic_parts.append('旅游 探险 出行')
            
        # 4. 清洗后的原始描述（保留关键信息）
        desc_snippet = cleaned_desc[:150] if cleaned_desc else ""
        
        # 组合所有部分
        embedding_parts = []
        embedding_parts.extend(core_attrs)
        if feature_text:
            embedding_parts.append(feature_text)
        if semantic_parts:
            embedding_parts.append(' '.join(semantic_parts))
        if desc_snippet:
            embedding_parts.append(desc_snippet)
            
        return ' '.join(embedding_parts)

    def process_products(self, products: List[Dict]) -> List[CleanedProduct]:
        """处理商品列表"""
        cleaned_products = []
        
        for product in products:
            try:
                title = product.get('title', '')
                desc = product.get('full_description', '')
                full_text = title + ' ' + desc
                
                brand = self.extract_brand(full_text)
                raw_model = self.extract_model(full_text)
                model = self.clean_model(raw_model)
                product_type = self.extract_product_type(full_text)
                color = self.extract_color(full_text)
                size, capacity = self.extract_size_capacity(full_text)
                features = self.extract_key_features(full_text)
                embedding_text = self.generate_embedding_text(product)
                
                # 确保价格是数值类型
                try:
                    price = float(product.get('price', 0))
                except (ValueError, TypeError):
                    price = 0.0
                
                cleaned_product = CleanedProduct(
                    id=product.get('id', ''),
                    brand=brand,
                    model=model,
                    product_type=product_type,
                    color=color,
                    size=size,
                    capacity=capacity,
                    price=price,
                    key_features=features,
                    embedding_text=embedding_text
                )
                
                cleaned_products.append(cleaned_product)
                
            except Exception as e:
                print(f"处理商品 {product.get('id', 'unknown')} 时出错: {e}")
                continue
        
        return cleaned_products

    def clean_model(self, model: str) -> str:
        """清理型号，去除重复和无关信息"""
        if not model:
            return ""
            
        # 移除中文字符和多余符号
        model = re.sub(r'[\u4e00-\u9fff]+', '', model)
        model = re.sub(r'[^\w]', '', model)
        
        # 提取核心型号
        core_patterns = [
            r'4QU[ET][A-Z0-9]{3,5}',
            r'23\d{3}',
            r'T50\d{1,3}[A-Z]?',
            r'TYP\d{1,3}[A-Z]?',
            r'JS00T\d{3}'
        ]
        
        for pattern in core_patterns:
            matches = re.findall(pattern, model, re.IGNORECASE)
            if matches:
                return matches[0].upper()
        
        # 如果没有匹配到标准格式，返回清理后的短字符串
        if len(model) > 0:
            return model[:10].upper()
            
        return ""

    def get_cleaning_stats(self, cleaned_products: List[CleanedProduct]) -> Dict[str, any]:
        """获取清洗统计信息"""
        total = len(cleaned_products)
        if total == 0:
            return {}
            
        stats = {
            "总商品数": total,
            "有品牌信息": len([p for p in cleaned_products if p.brand != "未知品牌"]),
            "有型号信息": len([p for p in cleaned_products if p.model]),
            "有颜色信息": len([p for p in cleaned_products if p.color]),
            "有尺寸信息": len([p for p in cleaned_products if p.size or p.capacity]),
            "有特征信息": len([p for p in cleaned_products if p.key_features]),
            "品牌分布": {},
            "类型分布": {},
            "价格范围": {}
        }
        
        # 品牌分布
        brand_count = {}
        for product in cleaned_products:
            brand_count[product.brand] = brand_count.get(product.brand, 0) + 1
        stats["品牌分布"] = dict(sorted(brand_count.items(), key=lambda x: x[1], reverse=True))
        
        # 类型分布
        type_count = {}
        for product in cleaned_products:
            type_count[product.product_type] = type_count.get(product.product_type, 0) + 1
        stats["类型分布"] = dict(sorted(type_count.items(), key=lambda x: x[1], reverse=True))
        
        # 价格统计
        prices = [p.price for p in cleaned_products if p.price > 0]
        if prices:
            stats["价格范围"] = {
                "最低价": min(prices),
                "最高价": max(prices),
                "平均价": round(sum(prices) / len(prices), 2),
                "中位数": sorted(prices)[len(prices)//2]
            }
        
        return stats

# 使用示例
def main():
    # 读取原始数据
    try:
        with open('products.json', 'r', encoding='utf-8') as f:
            products = json.load(f)
    except FileNotFoundError:
        print("未找到 products.json 文件")
        return
    
    # 初始化清洗器
    cleaner = ProductDataCleaner()
    
    # 处理数据
    cleaned_products = cleaner.process_products(products)
    
    # 输出示例结果
    print("=== 清洗结果预览 ===")
    for i, product in enumerate(cleaned_products[:3]):  # 只显示前3个
        print(f"\n=== 商品 {i+1} ===")
        print(f"ID: {product.id}")
        print(f"品牌: {product.brand}")
        print(f"型号: {product.model}")
        print(f"类型: {product.product_type}")
        print(f"颜色: {product.color}")
        print(f"尺寸: {product.size}")
        print(f"容量: {product.capacity}")
        print(f"价格: {product.price}元")
        print(f"特征: {product.key_features}")
        print(f"Embedding文本: {product.embedding_text[:200]}...")
    
    # 显示清洗统计
    stats = cleaner.get_cleaning_stats(cleaned_products)
    print(f"\n=== 清洗统计 ===")
    for key, value in stats.items():
        if isinstance(value, dict) and key in ["品牌分布", "类型分布"]:
            print(f"{key}: {dict(list(value.items())[:5])}")  # 只显示前5项
        else:
            print(f"{key}: {value}")
    
    # 保存清洗后的数据
    output_data = []
    for product in cleaned_products:
        output_data.append({
            'id': product.id,
            'brand': product.brand,
            'model': product.model,
            'product_type': product.product_type,
            'color': product.color,
            'size': product.size,
            'capacity': product.capacity,
            'price': product.price,
            'key_features': product.key_features,
            'embedding_text': product.embedding_text
        })
    
    try:
        with open('data/cleaned_product.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n清洗完成！共处理 {len(cleaned_products)} 个商品")
        print("数据已保存到 data/cleaned_product.json")
    except Exception as e:
        print(f"保存文件时出错: {e}")

if __name__ == "__main__":
    main()