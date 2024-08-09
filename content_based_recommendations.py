from db_connect import dbConnect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import re
from sklearn.preprocessing import normalize

# Kết nối đến cơ sở dữ liệu
db_connection = dbConnect()

# Kiểm tra xem kết nối đã được thiết lập thành công hay không
if db_connection.connection.is_connected():
    print("Kết nối đến cơ sở dữ liệu thành công")
else:
    print("Không thể kết nối đến cơ sở dữ liệu")

# Truy vấn để lấy dữ liệu từ bảng products
query = """
    SELECT p.id, p.name as product_name, p.des, b.name as brand_name, c.name as category_name, sc.name as sub_category_name
    FROM products p
    LEFT JOIN brands b ON p.brand_id = b.id
    LEFT JOIN categories c ON p.category_id = c.id
    LEFT JOIN sub_categories sc ON p.sub_category_id = sc.id
"""
df_products = pd.read_sql(query, db_connection.connection)

# Đóng kết nối
db_connection.close_connection()

# Hàm để làm sạch HTML và văn bản
def clean_text(raw_html):
    if raw_html is None:
        return ""
    # Loại bỏ HTML
    clean_text = BeautifulSoup(str(raw_html), "html.parser").get_text()
    # Chuyển về chữ thường
    clean_text = clean_text.lower()
    # Loại bỏ các ký tự đặc biệt và số
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    # Loại bỏ khoảng trắng thừa
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text

# Xử lý các giá trị null hoặc NaN
df_products = df_products.fillna('')

# Áp dụng hàm làm sạch cho các cột cần thiết
columns_to_clean = ['product_name', 'des', 'category_name', 'sub_category_name', 'brand_name']

for col in columns_to_clean:
    df_products[col] = df_products[col].astype(str).apply(clean_text)

# Loại bỏ các sản phẩm trùng lặp (nếu có)
df_products = df_products.drop_duplicates(subset='id')

# Gợi ý dựa trên nội dung
def content_based_recommendations(product_id, num_recommendations=3):
    tf = TfidfVectorizer(
        stop_words='english', 
        max_df=0.8,
        min_df=2,
        ngram_range=(1, 2),
        token_pattern=r'\b\w+\b'
    )
    
    df_products['content'] = (
        df_products['product_name'] + ' ' + 
        df_products['des'] + ' ' + 
        df_products['category_name'] + ' ' + 
        df_products['brand_name']
    )
    
    tf_matrix = tf.fit_transform(df_products['content'])
    cosine_sim = cosine_similarity(tf_matrix, tf_matrix)
    cosine_sim = normalize(cosine_sim)
    
    idx = df_products.index[df_products['id'] == product_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    product_indices = [i[0] for i in sim_scores]
    
    return df_products.iloc[product_indices]

# print("Content-based Recommendations:")
# print(content_based_recommendations(30))

# Endpoint API để lấy gợi ý sản phẩm
app = Flask(__name__)

@app.route('/recommendations', methods=['GET'])
def recommendations():
    product_id = int(request.args.get('product_id'))
    recommended_products = content_based_recommendations(product_id)
    recommended_product_ids = recommended_products['id'].tolist()
    data = {'recommended_product_ids': recommended_product_ids}
    return jsonify(data)

if __name__ == '__main__':
    app.run(port=9090)