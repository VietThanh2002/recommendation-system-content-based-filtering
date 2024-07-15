from db_connect import dbConnect


# Tạo một thể hiện của lớp dbConnect
db = dbConnect()

# Thực hiện truy vấn
query = """
    SELECT p.id, p.name, p.des, p.category_id, p.sub_category_id, p.brand_id, 
        b.name as brand_name, c.name as category_name, sc.name as sub_category_name
    FROM products p
    LEFT JOIN brands b ON p.brand_id = b.id
    LEFT JOIN categories c ON p.category_id = c.id
    LEFT JOIN sub_categories sc ON p.sub_category_id = sc.id
"""

db.execute_query(query)

# db.execute_query(query)