from db_connect import dbConnect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import re
from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

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

columns_to_clean = ['product_name', 'des', 'short_des', 'category_name', 'sub_category_name', 'brand_name']

for col in columns_to_clean:
    df_products[col] = df_products[col].astype(str).apply(clean_text)

# Loại bỏ các sản phẩm trùng lặp (nếu có)
df_products = df_products.drop_duplicates(subset='id')

# In ra cột thứ 7 (sub_category_name)
# print(df_products.iloc[:, 1])
# pd.set_option('display.max_columns', None)
# print(df_products)

# Gợi ý dựa trên nội dung
def content_based_recommendations(product_id,  num_recommendations=4):
    
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
        token_pattern=r'\b\w+\b',
    )
    
    df_products['features'] = (
        df_products['product_name'] + ' ' + 
        df_products['short_des'] + ' ' +
        df_products['des'] + ' ' + 
        df_products['category_name'] + ' ' +
        df_products['sub_category_name'] + ' ' + 
        df_products['brand_name']
    )    
    # print(df_products['features'])
   
    tf_idf_matrix = tf_idf_vectorizer.fit_transform(df_products['features'])
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # print(tf_idf_matrix)
    
    
    cosine_sim = cosine_similarity(tf_idf_matrix, tf_idf_matrix)
    # pd.set_option('display.max_rows', cosine_sim.shape[0])
    
    # pd.set_option('display.max_columns', cosine_sim.shape[1])
    # print(pd.DataFrame(cosine_sim).head(7))
    # print(cosine_sim)
    # cosine_sim = normalize(cosine_sim)
#    idx = product_id
#     print(idx)
#     sim_scores = list(enumerate(cosine_sim[product_id]))
    idx = df_products[df_products['id'] == product_id].index[0]
    print(idx)
    print(df_products[df_products['id'] == 30])
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    # print(sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # print(sim_scores)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    product_indices = [i[0] for i in sim_scores]
    
    print(product_indices)
    recommended_product_ids = df_products.iloc[product_indices]['id'].tolist()
    print(recommended_product_ids) 
     
    
    # print(product_indices)
    return df_products.iloc[product_indices]


# print("Content-based Recommendations:")
print('Ma trận độ tương đồng:')
print(content_based_recommendations(30))

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

# def evaluate_model_by_neighbors(df_products, content_based_recommendations, n_neighbors_list):
#     """
#     Đánh giá các chỉ số độ chính xác của mô hình dựa trên số lượng láng giềng gần nhất.
    
#         :param df_products: DataFrame chứa thông tin sản phẩm
#         :param content_based_recommendations: Hàm gợi ý dựa trên nội dung
#         :param n_neighbors_list: Danh sách số lượng láng giềng cần đánh giá
#         :return: DataFrame chứa độ chính xác, Precision, Recall, và F1-score cho mỗi số lượng láng giềng
#     """
#     # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
#     train_df, test_df = train_test_split(df_products, test_size=0.2, random_state=42, shuffle=True)
#     # print(train_df)
#     # print(test_df)
    
#     results = []

#     for n_neighbors in n_neighbors_list:
#         y_true = []
#         y_pred = []
        
#         for _, product in test_df.iterrows():
#             true_category = product['category_name']
#             print(product['id'])
        
#             # Lấy gợi ý dựa trên nội dung
#             recommendations = content_based_recommendations(product['id'], num_recommendations=n_neighbors)
            
#             # Lấy danh mục phổ biến nhất trong các gợi ý
#             predicted_category = recommendations['category_name'].mode().iloc[0]
#             print(predicted_category)
            
#             y_true.append(true_category)
#             y_pred.append(predicted_category)
            
#             # print('Confusion Matrix:')
#             # print(pd.crosstab(pd.Series(y_true), pd.Series(y_pred), rownames=['True'], colnames=['Predicted'], margins=True))
#         # Tính toán các chỉ số
        
#         accuracy = accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
#         recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
#         f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
#         # Lưu kết quả cho mỗi số lượng láng giềng
#         results.append({
#             'n_neighbors': n_neighbors,
#             'accuracy': accuracy * 100,  # Chuyển thành phần trăm
#             'precision': precision * 100,
#             'recall': recall * 100,
#             'f1_score': f1 * 100
#         })
    
#     return pd.DataFrame(results)

# # Sử dụng
# n_neighbors_list = [5, 10, 20, 30, 40, 50, 60]
# evaluation_results = evaluate_model_by_neighbors(df_products, content_based_recommendations, n_neighbors_list)

# # In kết quả
# print(evaluation_results)
# # Đảm bảo các giá trị cột accuracy đã tồn tại
# if 'accuracy' not in evaluation_results.columns:
#     print("Cột 'accuracy' không tồn tại trong dữ liệu kết quả.")
# else:
#     print("Cột 'accuracy' đã có trong kết quả.")
# # Vẽ đồ thị các chỉ số

# # Dùng vòng lặp để vẽ các chỉ số
# plt.figure(figsize=(12, 8))
# metrics = ['accuracy', 'precision', 'recall', 'f1_score']
# colors = ['b', 'r', 'g', 'c']
# markers = ['o', 's', '^', 'D']
# linestyles = ['-', '--', '-.', ':']

# for i, metric in enumerate(metrics):
#     plt.plot(evaluation_results['n_neighbors'], evaluation_results[metric], 
#              color=colors[i], marker=markers[i], linestyle=linestyles[i],
#              label=metric, linewidth=2, markersize=8, alpha=0.7)
    
#      # Hiển thị giá trị các chỉ số ngay tại mỗi điểm
#     for i, value in enumerate(evaluation_results[metric]):
#         plt.text(evaluation_results['n_neighbors'][i], value, f'{value:.2f}', fontsize=9, ha='right')

# plt.title('Model Performance Metrics')
# plt.xlabel('Số Láng Giềng Gần Nhất')
# plt.ylabel('Điểm Số (%)')
# plt.legend()
# plt.grid(True)
# plt.show()


# # Phân tích phân bố danh mục
# category_distribution = df_products['category_name'].value_counts(normalize=True)
# plt.figure(figsize=(12, 6))
# category_distribution.plot(kind='bar')
# plt.title('Phân Bố Danh Mục')
# plt.xlabel('Danh Mục')
# plt.ylabel('Tỷ Lệ')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

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