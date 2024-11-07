
from content_based_recommendations import content_based_recommendations, df_products
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def evaluate_model(df_products, content_based_recommendations, n_neighbors_list):
    # pd.set_option('display.max_columns', df_products.shape[1])
    # print(df_products)
    results = []

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    train_df, test_df = train_test_split(df_products, test_size=0.2, random_state=42, shuffle=True)
    
    print('Train data:', train_df)
    print('------------------------------------')
    print('Test data:', test_df)
    print('------------------------------------')
    
    for n_neighbors in n_neighbors_list:
        y_true = []
        y_pred = []

        # Duyệt qua từng sản phẩm trong tập kiểm tra
        for _, product in test_df.iterrows():
        
            true_category = product['category_name']
            # print('n_neighbors:', n_neighbors)
            # print('True category:', true_category)
            # print(true_category)
            
            # Lấy gợi ý dựa trên nội dung
            # product_id = product['id'] 
            # product_name = product['product_name']
            # print('Product ID:', product_id)
            # print('Product Name:', product_name)
            recommendations = content_based_recommendations(product['id'], num_recommendations=n_neighbors)
            # print('Recommendations:', recommendations)
            
            # Lấy danh mục phổ biến nhất trong các gợi ý
            predicted_category = recommendations['category_name'].mode().iloc[0]
            # print('Predicted category:', predicted_category)
            # print(predicted_category)
            
            y_true.append(true_category)
            # print('y_true:', y_true)
            y_pred.append(predicted_category)
            # print('y_pred:', y_pred)
            
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Lưu kết quả cho mỗi số lượng láng giềng
        results.append({
            'n_neighbors': n_neighbors,
            'accuracy': accuracy * 100,  # Chuyển thành phần trăm
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100
        })
    
    results_df = pd.DataFrame(results)

    # Vẽ đồ thị các chỉ số
    plt.figure(figsize=(12, 8))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = ['b', 'r', 'g', 'c']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']

    for i, metric in enumerate(metrics):
        plt.plot(results_df['n_neighbors'], results_df[metric], 
                 color=colors[i], marker=markers[i], linestyle=linestyles[i],
                 label=metric, linewidth=2, markersize=8, alpha=0.7)
        
        # Hiển thị giá trị các chỉ số ngay tại mỗi điểm
        for j, value in enumerate(results_df[metric]):
            plt.text(results_df['n_neighbors'][j], value, f'{value:.2f}', fontsize=9, ha='right')

    plt.title('Model Performance Metrics')
    plt.xlabel('Số Láng Giềng Gần Nhất')
    plt.ylabel('Điểm Số (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return results_df

# Sử dụng hàm
n_neighbors_list = [5, 10, 15, 20, 25]
# n_neighbors_list = [5]
evaluation_results = evaluate_model(df_products, content_based_recommendations, n_neighbors_list)

# In kết quả
print(evaluation_results)


# Phân tích phân bố danh mục
category_distribution = df_products['category_name'].value_counts(normalize=True)
plt.figure(figsize=(12, 6))
category_distribution.plot(kind='bar')
plt.title('Phân Bố Danh Mục')
plt.xlabel('Danh Mục')
plt.ylabel('Tỷ Lệ')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()