import mysql.connector
from mysql.connector import Error

class dbConnect:
    def __init__(self):
        self.connection = mysql.connector.connect(
            host='localhost',
            database='project_ct550',
            user='root',
            password=''
        )
        if self.connection.is_connected():
            print("Kết nối thành công đến cơ sở dữ liệu")
        else:
            print("Không thể kết nối đến cơ sở dữ liệu")
    
    def execute_query(self, query, params=None):
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            result = cursor.fetchall()  # Lấy tất cả các dòng kết quả
            for row in result:
                print(row)  # In từng dòng kết quả ra màn hình
            self.connection.commit()
            print("Truy vấn được thực thi thành công")
            return result  # Trả về kết quả của truy vấn
        except Error as e:
            print(f"Lỗi khi thực thi truy vấn: {e}")
     
            
    def close_connection(self):
        if self.connection.is_connected():
            self.connection.close()
            print("Đóng kết nối đến cơ sở dữ liệu") 