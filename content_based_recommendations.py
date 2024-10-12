from db_connect import dbConnect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import re
from sklearn.preprocessing import normalize

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Vẽ đồ thị
import matplotlib.pyplot as plt

# Kết nối đến cơ sở dữ liệu
db_connection = dbConnect()

# Kiểm tra xem kết nối đã được thiết lập thành công hay không
if db_connection.connection.is_connected():
    print("Kết nối đến cơ sở dữ liệu thành công")
else:
    print("Không thể kết nối đến cơ sở dữ liệu")

# Truy vấn để lấy dữ liệu từ bảng products
query = """
    SELECT p.id, p.name as product_name, p.des, p.short_des as short_des, b.name as brand_name, c.name as category_name, sc.name as sub_category_name
    FROM products p
    LEFT JOIN brands b ON p.brand_id = b.id
    LEFT JOIN categories c ON p.category_id = c.id
    LEFT JOIN sub_categories sc ON p.sub_category_id = sc.id
"""
df_products = pd.read_sql(query, db_connection.connection)

# print(df_products)
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
columns_to_clean = ['product_name', 'des', 'short_des', 'category_name', 'sub_category_name', 'brand_name']

for col in columns_to_clean:
    df_products[col] = df_products[col].astype(str).apply(clean_text)

# Loại bỏ các sản phẩm trùng lặp (nếu có)
df_products = df_products.drop_duplicates(subset='id')

# pd.set_option('display.max_columns', None)
# print(df_products)

# Gợi ý dựa trên nội dung
def content_based_recommendations(product_id, num_recommendations=5):
    
    stop_words_vi = [
        'các', 'và', 'là', 'của', 'trong', 
        'với', 'đến', 'cho', 'nhưng', 'không', 
        'được', 'một', 'nhiều', 'hơn', 'thế', 
        'này', 'cái', 'vì', 'hoặc', 'tại'
    ]   
    
    tf_idf_vectorizer = TfidfVectorizer(
        stop_words=stop_words_vi, 
        max_df=0.8,
        min_df=2,
        ngram_range=(1, 2),
        token_pattern=r'\b\w+\b'
    )
    
    df_products['features'] = (
        df_products['product_name'] + ' ' + 
        df_products['short_des'] + ' ' +
        df_products['des'] + ' ' + 
        df_products['category_name'] + ' ' +
        df_products['sub_category_name'] + ' ' + 
        df_products['brand_name']
    )    
   
    tf_idf_matrix = tf_idf_vectorizer.fit_transform(df_products['features'])
    # pd.set_option('display.max_columns', None)
    # print(tf_matrix)
    
    
    cosine_sim = cosine_similarity(tf_idf_matrix, tf_idf_matrix)
    
    # pd.set_option('display.max_rows', cosine_sim.shape[0])
    
    # # pd.set_option('display.max_columns', cosine_sim.shape[1])
    # print(pd.DataFrame(cosine_sim).head(7))
    # print(cosine_sim)
    # cosine_sim = normalize(cosine_sim)
#    idx = product_id
#     print(idx)
#     sim_scores = list(enumerate(cosine_sim[product_id]))
    idx = df_products[df_products['id'] == product_id].index[0]
    # print(idx)
    # print(df_products[df_products['id'] == 30])
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # print(sim_scores)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    product_indices = [i[0] for i in sim_scores]
    
    # print(product_indices)
    
    return df_products.iloc[product_indices]


# print("Content-based Recommendations:")
print('Ma trận độ tương đồng:')
print(content_based_recommendations(30))



def evaluate_accuracy_by_neighbors(df_products, content_based_recommendations, n_neighbors_list):
    """
    Đánh giá độ chính xác của mô hình dựa trên số lượng láng giềng gần nhất.
    
    :param df_products: DataFrame chứa thông tin sản phẩm
    :param content_based_recommendations: Hàm gợi ý dựa trên nội dung
    :param n_neighbors_list: Danh sách số lượng láng giềng cần đánh giá
    :return: Dictionary chứa độ chính xác cho mỗi số lượng láng giềng
    """
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    train_df, test_df = train_test_split(df_products, test_size=0.2, random_state=42)
    
    accuracy_results = {}
    
    for n_neighbors in n_neighbors_list:
        y_true = []
        y_pred = []
        
        for _, product in test_df.iterrows():
            true_category = product['category_name']
            
            # Lấy gợi ý dựa trên nội dung
            recommendations = content_based_recommendations(product['id'], num_recommendations=n_neighbors)
            
            # Lấy danh mục phổ biến nhất trong các gợi ý
            predicted_category = recommendations['category_name'].mode().iloc[0]
            
            y_true.append(true_category)
            y_pred.append(predicted_category)
        
        # Tính độ chính xác
        accuracy = accuracy_score(y_true, y_pred)
        accuracy_results[n_neighbors] = accuracy * 100  # Chuyển đổi thành phần trăm
    
    return accuracy_results

# Ví dụ sử dụng
n_neighbors_list = [5, 10, 20, 30, 40, 50]
accuracy_results = evaluate_accuracy_by_neighbors(df_products, content_based_recommendations, n_neighbors_list)

# In kết quả
for n, acc in accuracy_results.items():
    print(f"Số láng giềng: {n}, Độ chính xác: {acc:.2f}%")

# Vẽ đồ thị (nếu cần)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(list(accuracy_results.keys()), list(accuracy_results.values()), marker='o')
plt.title('Độ chính xác theo số lượng láng giềng gần nhất')
plt.xlabel('Số lượng láng giềng gần nhất')
plt.ylabel('Độ chính xác (%)')
plt.grid(True)
plt.show()


# Endpoint API để lấy gợi ý sản phẩm
# app = Flask(__name__)

# @app.route('/recommendations', methods=['GET'])
# def recommendations():
#     product_id = int(request.args.get('product_id'))
#     recommended_products = content_based_recommendations(product_id)
#     recommended_product_ids = recommended_products['id'].tolist()
#     data = {'recommended_product_ids': recommended_product_ids}
#     return jsonify(data)

# if __name__ == '__main__':
#     app.run(port=9090)