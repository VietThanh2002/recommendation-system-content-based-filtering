from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Giả sử chúng ta có một DataFrame chứa thông tin sản phẩm và dữ liệu đánh giá của người dùng
df_products = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
    'description': ['desc A', 'desc B', 'desc C', 'desc D', 'desc E'],
    'category': ['cat1', 'cat2', 'cat1', 'cat2', 'cat1']
})

df_ratings = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
    'product_id': [1, 2, 2, 3, 3, 4, 4, 5],
    'rating': [5, 4, 4, 5, 5, 3, 2, 4]
})

# Gợi ý dựa trên nội dung
def content_based_recommendations(product_id, num_recommendations=5):
    tf = TfidfVectorizer(stop_words='english')
    tf_matrix = tf.fit_transform(df_products['description'] + ' ' + df_products['category'])
    cosine_sim = cosine_similarity(tf_matrix, tf_matrix)
    
    idx = df_products.index[df_products['id'] == product_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    product_indices = [i[0] for i in sim_scores]
    return df_products.iloc[product_indices]

# Gợi ý dựa trên cộng tác
def collaborative_filtering_recommendations(user_id, num_recommendations=5):
    user_ratings = df_ratings.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
    user_sim = cosine_similarity(user_ratings)
    
    idx = user_ratings.index.get_loc(user_id)
    sim_scores = list(enumerate(user_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    similar_users = [i[0] for i in sim_scores]
    similar_users_ratings = user_ratings.iloc[similar_users].mean(axis=0)
    similar_users_ratings = similar_users_ratings.sort_values(ascending=False)
    
    return df_products[df_products['id'].isin(similar_users_ratings.index[:num_recommendations])]

# Kết hợp hai phương pháp
def hybrid_recommendations(user_id, product_id, num_recommendations=5):
    content_recommendations = content_based_recommendations(product_id, num_recommendations)
    collab_recommendations = collaborative_filtering_recommendations(user_id, num_recommendations)
    
    combined_recommendations = pd.concat([content_recommendations, collab_recommendations]).drop_duplicates().head(num_recommendations)
    return combined_recommendations

# Ví dụ sử dụng
print("Content-based Recommendations:")
print(content_based_recommendations(1))

print("Collaborative Filtering Recommendations:")
print(collaborative_filtering_recommendations(1))

print("Hybrid Recommendations:")
print(hybrid_recommendations(1, 1))
