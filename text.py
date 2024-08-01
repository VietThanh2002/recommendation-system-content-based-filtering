from db_connect import dbConnect
import re
from bs4 import BeautifulSoup

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

# Tạo một thể hiện của lớp dbConnect
db = dbConnect()

# Thực hiện truy vấn
query = """
    SELECT p.id, p.name, p.des, b.name as brand_name, c.name as category_name, sc.name as sub_category_name
    FROM products p
    LEFT JOIN brands b ON p.brand_id = b.id
    LEFT JOIN categories c ON p.category_id = c.id
    LEFT JOIN sub_categories sc ON p.sub_category_id = sc.id
"""

results = db.execute_query(query)

# Lấy tên các cột
columns = ['id', 'name', 'des', 'brand_name', 'category_name', 'sub_category_name']

# Làm sạch dữ liệu và in ra từng hàng
for row in results:
    cleaned_row = {
        'id': row[0],
        'name': clean_text(row[1]),
        'des': clean_text(row[2]),
        'brand_name': clean_text(row[3]),
        'category_name': clean_text(row[4]),
        'sub_category_name': clean_text(row[5])
    }
    
    print("Cleaned Row:")
    
    for key, value in cleaned_row.items():
        print(f"{key}: {value}")
    print("------------------------")

# Đóng kết nối
db.close_connection()