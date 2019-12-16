import logging
import pymysql.cursors
import numpy as np
from tqdm import tqdm


class mysql_client:
    def __init__(self, host="localhost", user="root", password="example", db="yelp", charset="utf8"):
        self.conn = pymysql.connect(
            host=host,
            port=3306,
            user=user,
            password=password,
            db=db,
        )

        self.total_businesses = None
        self.total_users = None

        self.train_start_seq = 0
        self.train_end_seq = None

        self.validation_start_seq = None
        self.validation_end_seq = None

        self.user_index = {}
        self.business_index = {}
        self.index2business = {}

    def init(self):

        logging.info("Retrieval user index")
        sql = "SELECT * FROM users"
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            users = cursor.fetchall()
            for idx, user in enumerate(users):
                self.user_index[idx] = user[0]

            self.total_users = len(self.user_index)
            logging.info(f"the number of total users: {self.total_users}")

        logging.info("Retrieval business index")
        sql = "SELECT * FROM business_ids"
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            business_ids = cursor.fetchall()

            for idx, business_id in enumerate(business_ids):
                self.business_index[business_id[0]] = idx
                self.index2business[idx] = business_id[0]

            self.total_businesses = len(self.business_index)

        self.train_end_seq = int(np.floor(self.total_users * 0.7))
        self.validation_start_seq = self.train_end_seq + 1
        self.validation_end_seq = self.total_users - 1

    def __retrival_businesses(self, user_id):
        sql = f"SELECT business_id FROM data WHERE user_id='{user_id}'"
        businesses = []

        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()

            for elem in results:
                businesses.append(elem[0])
            return businesses

    def __retrival_userIds(self, start_user_seq, end_user_seq):
        user_ids = []
        for user_seq in range(start_user_seq, end_user_seq):
            user_ids.append(self.user_index[user_seq])
        return user_ids

    def get_data(self, start_user_seq, end_user_seq):
        user_ids = self.__retrival_userIds(start_user_seq, end_user_seq)

        X = []
        Y = []
        for user_id in user_ids:
            businesses_ids = self.__retrival_businesses(user_id)
            selected = np.random.choice(businesses_ids, 2, replace=False)

            x = self.business_index[selected[0]]
            y = self.business_index[selected[1]]

            assert (x != y)

            X.append(x)
            Y.append(y)

        return X, Y


if __name__ is '__main__':
    log_format = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    client = mysql_client()
    client.init()
    client.get_data(0, 100)

# user_id = client.retrival_userId(1)
# businesses = client.retrival_businesses(user_id)

# print(businesses)
