from tensorflow.keras.utils import Sequence
import numpy as np


class DataGenerator(Sequence):
    def __init__(self, start_seq, end_seq, mysql_client, batch_size=100):
        self.mysql_client = mysql_client
        self.batch_size = batch_size

        self.total_users = end_seq - start_seq
        self.start_seq = start_seq
        self.end_seq = end_seq
        self.total_businesses = mysql_client.total_businesses

    def __len__(self):
        batch_size = int(np.floor(self.total_users / self.batch_size))
        return batch_size

    def __getitem__(self, index):
        start_user_seq = self.start_seq + (index * self.batch_size)
        end_user_seq = start_user_seq + self.batch_size

        return self.__data_generation(start_user_seq, end_user_seq)

    def __data_generation(self, start_user_seq, end_user_seq):
        business_input, business_target = self.mysql_client.get_data(start_user_seq, end_user_seq)

        business_input = np.array(business_input, dtype=int)

        num = np.random.randint(3)

        if num % 3 > 0:
            business_target = np.array(business_target, dtype=int)
            y = np.ones(100)
        else:
            negative_samples = np.random.randint(0, self.total_businesses, size=100)
            business_target = np.array(negative_samples, dtype=int)
            y = np.zeros(100)

        return {'business_input': business_input, 'business_target': business_target}, y




