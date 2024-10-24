
from content_based_recommendations import content_based_recommendations, df_products

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def evaluate_model(df_products, content_based_recommendations, n_neighbors_list):
    results = []

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    train_df, test_df = train_test_split(df_products, test_size=0.2, random_state=42, shuffle=True)

    for n_neighbors in n_neighbors_list:
        y_true = []
        y_pred = []

        # Duyệt qua từng sản phẩm trong tập kiểm tra
        for _, product in test_df.iterrows():
            true_category = product['category_name']
            
            # Lấy gợi ý dựa trên nội dung
            recommendations = content_based_recommendations(product['id'], num_recommendations=n_neighbors)
            
            # Lấy danh mục phổ biến nhất trong các gợi ý
            predicted_category = recommendations['category_name'].mode().iloc[0]
            
            y_true.append(true_category)
            y_pred.append(predicted_category)

        # Tính toán các chỉ số
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
n_neighbors_list = [5, 10, 20, 30, 40, 50, 60]
evaluation_results = evaluate_model(df_products, content_based_recommendations, n_neighbors_list)

# In kết quả
print(evaluation_results)