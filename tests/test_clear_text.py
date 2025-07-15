import re
from typing import Dict, List
import json

class DataQualityAnalyzer:
    """数据质量分析器"""
    
    def __init__(self):
        self.issues = []
        self.suggestions = []
    
    def analyze_model_extraction(self, products: List[Dict]) -> Dict:
        """分析型号提取质量"""
        issues = {
            'too_long': [],  # 型号过长
            'empty': [],     # 型号为空
            'contains_chinese': [],  # 包含中文
            'duplicate_info': []     # 包含重复信息
        }
        
        for product in products:
            model = product.get('model', '')
            product_id = product.get('id')
            
            # 检查型号为空
            if not model:
                issues['empty'].append(product_id)
            
            # 检查型号过长（超过20字符可能有问题）
            elif len(model) > 20:
                issues['too_long'].append({
                    'id': product_id,
                    'model': model,
                    'length': len(model)
                })
            
            # 检查是否包含中文
            if re.search(r'[\u4e00-\u9fff]', model):
                issues['contains_chinese'].append({
                    'id': product_id,
                    'model': model
                })
            
            # 检查是否包含重复信息
            if model and (model.lower() in product.get('embedding_text', '').lower()):
                duplicate_count = product['embedding_text'].lower().count(model.lower())
                if duplicate_count > 2:  # 出现超过2次认为重复
                    issues['duplicate_info'].append({
                        'id': product_id,
                        'model': model,
                        'count': duplicate_count
                    })
        
        return issues
    
    def suggest_model_improvements(self, products: List[Dict]) -> List[Dict]:
        """建议型号改进方案"""
        improved_products = []
        
        for product in products.copy():
            original_model = product.get('model', '')
            improved_model = self.clean_model(original_model)
            
            if improved_model != original_model:
                product['model'] = improved_model
                product['_model_changed'] = True
                product['_original_model'] = original_model
            
            improved_products.append(product)
        
        return improved_products
    
    def clean_model(self, model: str) -> str:
        """清洗型号字段"""
        if not model:
            return ""
        
        # 移除中文描述
        model = re.sub(r'[\u4e00-\u9fff]+', '', model)
        
        # 提取核心型号模式
        patterns = [
            r'4QU[ET]\w{0,4}',  # JanSport 4QUE/4QUT系列
            r'23\d{3}',         # 北极狐23xxx系列  
            r'TYP\w{0,5}',      # JanSport TYP系列
            r'T\d{2,4}\w*',     # T系列
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, model, re.IGNORECASE)
            if matches:
                return matches[0].upper()
        
        # 如果没有匹配到标准格式，保留数字字母组合
        clean_model = re.sub(r'[^A-Za-z0-9]', '', model)
        return clean_model[:15] if clean_model else ""
    
    def analyze_embedding_text_quality(self, products: List[Dict]) -> Dict:
        """分析embedding文本质量"""
        issues = {
            'too_long': [],
            'too_short': [],
            'high_redundancy': [],
            'missing_key_info': []
        }
        
        for product in products:
            embedding_text = product.get('embedding_text', '')
            product_id = product.get('id')
            
            # 长度检查
            if len(embedding_text) > 500:
                issues['too_long'].append(product_id)
            elif len(embedding_text) < 50:
                issues['too_short'].append(product_id)
            
            # 冗余度检查（简单的重复词检查）
            words = embedding_text.split()
            unique_words = set(words)
            redundancy = 1 - len(unique_words) / len(words) if words else 0
            
            if redundancy > 0.3:  # 超过30%重复
                issues['high_redundancy'].append({
                    'id': product_id,
                    'redundancy': redundancy
                })
            
            # 关键信息缺失检查
            required_fields = ['品牌', '价格']
            missing_fields = []
            for field in required_fields:
                if field not in embedding_text:
                    missing_fields.append(field)
            
            if missing_fields:
                issues['missing_key_info'].append({
                    'id': product_id,
                    'missing': missing_fields
                })
        
        return issues

# 使用示例
def analyze_product_data(products_data):
    analyzer = DataQualityAnalyzer()
    
    print("=== 数据质量分析报告 ===\n")
    
    # 分析型号质量
    model_issues = analyzer.analyze_model_extraction(products_data)
    
    print("1. 型号字段问题:")
    print(f"   - 型号为空: {len(model_issues['empty'])} 个商品")
    print(f"   - 型号过长: {len(model_issues['too_long'])} 个商品")
    print(f"   - 包含中文: {len(model_issues['contains_chinese'])} 个商品")
    
    if model_issues['too_long']:
        print("\n   过长型号示例:")
        for item in model_issues['too_long'][:3]:
            print(f"   ID: {item['id']}, 型号: {item['model'][:50]}...")
    
    # 分析embedding文本质量
    embedding_issues = analyzer.analyze_embedding_text_quality(products_data)
    
    print(f"\n2. Embedding文本问题:")
    print(f"   - 文本过长: {len(embedding_issues['too_long'])} 个")
    print(f"   - 文本过短: {len(embedding_issues['too_short'])} 个")
    print(f"   - 冗余度高: {len(embedding_issues['high_redundancy'])} 个")
    
    # 生成改进建议
    print(f"\n3. 改进建议:")
    print("   ✅ 重新提取型号，使用更精确的正则表达式")
    print("   ✅ 优化embedding文本结构，减少冗余")
    print("   ✅ 标准化颜色描述")
    print("   ✅ 补充缺失的关键属性")
    
    return model_issues, embedding_issues

# 模拟分析（基于提供的数据样本）
try:
    # 从JSON文件加载数据
    with open('data/cleaned_product.json', 'r', encoding='utf-8') as f:
        products_data = json.load(f)
    print(f"成功加载了{len(products_data)}个商品数据")
except Exception as e:
    print(f"加载JSON文件失败: {e}")
    # 如果加载失败，使用示例数据
    products_data = [
        {
            "id": "940387428910",
            "model": "4QUE008JANSPORT绝版2017年迪士尼联名4QUE008",
            "embedding_text": "品牌：JanSport 型号：4QUE008JANSPORT绝版2017年迪士尼联名4QUE008 类型：背包 价格：138元"
        },
        {
            "id": "939867633438", 
            "model": "",
            "embedding_text": "品牌：Fjallraven 类型：双肩包 尺寸：中号 价格：102元"
        }
    ]

# 执行分析
model_issues, embedding_issues = analyze_product_data(products_data)