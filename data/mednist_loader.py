from core.DataLoader import DefaultDataset


class MedNISTDataset(DefaultDataset):
    def __init__(self, data_dir, file_type='', label_dir=None, target_size=(64, 64)):
        super(MedNISTDataset, self).__init__(data_dir, file_type, label_dir, target_size)

    def get_label(self, idx):
        path_name = self.files[idx]
        if 'CXR' in path_name:
            return 0
        elif 'Abdomen' in path_name:
            return 1
        elif 'Head' in path_name:
            return 2
        elif 'Hand' in path_name:
            return 3
        elif 'Breast' in path_name:
            return 4
        else:
            return 0