from db_connect import dbConnect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import re
from sklearn.preprocessing import normalize
# Tạo một thể hiện của lớp dbConnect
db_connection = dbConnect()

# Thực hiện truy vấn
query = """
    SELECT p.id, p.name as product_name, p.des, p.short_des, b.name as brand_name, c.name as category_name
    FROM products p
    LEFT JOIN brands b ON p.brand_id = b.id
    LEFT JOIN categories c ON p.category_id = c.id
"""

content_product = pd.read_sql(query, db_connection.connection)


db_connection.close_connection()

# Hàm để làm sạch HTML và văn bản
def clean_text(raw_text):
    if not isinstance(raw_text, str):
        return ""
    # Loại bỏ HTML
    clean_text = BeautifulSoup(raw_text, "html.parser").get_text()
    # Chuyển về chữ thường
    clean_text = clean_text.lower()
    # Loại bỏ các ký tự đặc biệt và số
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    # Loại bỏ khoảng trắng thừa
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text
# Lấy tên các cột
# Áp dụng hàm làm sạch cho các cột cần thiết
columns_to_clean = ['product_name', 'des', 'short_des', 'category_name', 'brand_name']

for col in columns_to_clean:
   content_product [col] = content_product [col].astype(str).apply(clean_text)
    
# content_product = {
#     'product_name': '',
#     'des': '',
#     'category_name': '',
#     'brand_name': ''
# }

content_product['content'] = (content_product['product_name'] + ' ' + content_product['product_name'] +' ' + content_product['category_name'] + ' ' + content_product['brand_name'])

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print("Content Product:")
print(content_product)
# for key, value in content_product.items():
#     print(f"{key}: {value}")
# print("------------------------")

# # Làm sạch dữ liệu và in ra từng hàng
# for row in results:
#     cleaned_row = {
#         'id': row[0],
#         'name': clean_text(row[1]),
#         'des': clean_text(row[2]),
#         'short_des': clean_text(row[3]),
#         'brand_name': clean_text(row[4]),
#         'category_name': clean_text(row[5])
#     }
    
#     print("Cleaned Row:")
    
#     for key, value in cleaned_row.items():
#         print(f"{key}: {value}")
#     print("------------------------")

# Đóng kết nối
# db.close_connection()