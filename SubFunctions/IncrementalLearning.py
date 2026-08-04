class Incremental_Learning:
    def __init__(self, train_data, train_labels, number_of_chunks=5):

        self.train_data = train_data
        self.train_labels = train_labels
        self.number_of_chunks = number_of_chunks

    @staticmethod
    def split_list(input_list, num_chunks):
        avg_chunk_size = len(input_list) // num_chunks
        remaining_elements = len(input_list) % num_chunks

        split_lists = []
        start = 0
        for i in range(num_chunks):
            chunk_size = avg_chunk_size + (1 if i < remaining_elements else 0)
            split_lists.append(input_list[start:start + chunk_size])
            start += chunk_size

        split_list = [split_lists[0], split_lists[0] + split_lists[1], split_lists[0] + split_lists[1] + split_lists[2],
                      split_lists[0] + split_lists[1] + split_lists[2] + split_lists[3],
                      split_lists[0] + split_lists[1] + split_lists[2] + split_lists[3] + split_lists[4]]

        return split_list

    def increment_data(self):
        idx = [_ for _ in range(len(self.train_data))]

        split_idx = self.split_list(idx, self.number_of_chunks)

        train_sets = []
        for i in range(len(split_idx)):
            index = split_idx[i]
            data = self.train_data[index]
            labels = self.train_labels[index]
            train_sets.append([data, labels])

        return train_sets
