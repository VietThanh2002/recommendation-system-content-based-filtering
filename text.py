from db_connect import dbConnect


# Tạo một thể hiện của lớp dbConnect
db = dbConnect()

# Thực hiện truy vấn
query = "SELECT id, name, slug FROM products"

db.execute_query(query)

# db.execute_query(query)