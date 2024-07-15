from db_connect import dbConnect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup

# Kết nối đến cơ sở dữ liệu
db_connection = dbConnect()

# Kiểm tra xem kết nối đã được thiết lập thành công hay không
if db_connection.connection.is_connected():
    print("Kết nối đến cơ sở dữ liệu thành công")
else:
    print("Không thể kết nối đến cơ sở dữ liệu")

# Truy vấn để lấy dữ liệu từ bảng products
query = """
    SELECT p.id, p.name, p.des, p.category_id, p.sub_category_id, p.brand_id, 
        b.name as brand_name, c.name as category_name, sc.name as sub_category_name
    FROM products p
    LEFT JOIN brands b ON p.brand_id = b.id
    LEFT JOIN categories c ON p.category_id = c.id
    LEFT JOIN sub_categories sc ON p.sub_category_id = sc.id
"""
df_products = pd.read_sql(query, db_connection.connection)

# Đóng kết nối
db_connection.close_connection()

# Hàm để làm sạch HTML
def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    clean_text = soup.get_text()
    return clean_text

# Áp dụng hàm làm sạch cho cột 'des'
df_products['clean_des'] = df_products['des'].apply(clean_html)

# Gợi ý dựa trên nội dung
def content_based_recommendations(product_id, num_recommendations=3):
    tf = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=5, ngram_range=(1, 2))
    df_products['content'] = (
        df_products['clean_des'] + ' ' + 
        df_products['category_name'].astype(str) + ' ' + 
        df_products['sub_category_name'].astype(str) + ' ' +
        df_products['brand_name'].astype(str)
    )
    tf_matrix = tf.fit_transform(df_products['content'])
    cosine_sim = cosine_similarity(tf_matrix, tf_matrix)
    # print(cosine_sim)
    
    idx = df_products.index[df_products['id'] == product_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    print(sim_scores)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    product_indices = [i[0] for i in sim_scores]
    return df_products.iloc[product_indices]

print("Content-based Recommendations:")
print(content_based_recommendations(30))
