import logging
import pymysql.cursors


class mysql_client:
    def __init__(self, host="localhost", user="root", password="example", db="yelp", charset="utf8"):
        self.conn = None
        self.host = host
        self.user = user
        self.password = password
        self.db = db

    def connect(self):
        self.conn = pymysql.connect(
            host=self.host,
            port=3306,
            user=self.user,
            password=self.password,
            db=self.db,
        )

    def retrival_businesses(self, user_id):
        sql = f"SELECT business_id FROM data WHERE user_id='{user_id}'"
        businesses = []

        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()

            for elem in results:
                businesses.append(elem[0])
            return businesses

    def retrival_userId(self, seq):
        sql = f"SELECT user_id FROM users WHERE seq={seq}"
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchone()
            return result[0]


client = mysql_client()
client.connect()

user_id = client.retrival_userId(1)
businesses = client.retrival_businesses(user_id)

print(businesses)
