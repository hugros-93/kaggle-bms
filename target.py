import pandas as pd

class TargetSet():
    def __init__(self, limit=None):
        self.limit = limit
        self.load_file()
        self.split_target()

    def load_file(self):
        self.targets = pd.read_csv("bms-molecular-translation/train_labels.csv")
        if self.limit:
            self.targets = self.targets[:self.limit]

    def split_target(self):
        self.targets['targets'] = self.targets['InChI'].apply(lambda x: x.split('/'))
        self.targets['target_A'] = self.targets['targets'].apply(lambda x: x[1])
        self.targets['target_B'] = self.targets['targets'].apply(lambda x: x[2])
        self.targets['target_C'] = self.targets['targets'].apply(lambda x: x[3] if len(x)>3 else '')
        del self.targets['targets']
        self.targets = self.targets.set_index("image_id")
        
    def get_targets(self):
        return self.targets.transpose().to_dict()
