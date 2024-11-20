from torch.utils.data import DataLoader

'''
class PriorDataLoader(DataLoader):
    pass
    # init accepts num_steps as first argument

    # has two attributes set on class or object level:
    # num_features: int and
    # num_outputs: int
    # fuse_x_y: bool
    # Optional: validate function that accepts a transformer model
'''

class PriorDataLoader(DataLoader):
    def __init__(self, *args, num_features=None, hyperparameters=None, batch_size_per_gp_sample=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = num_features
        self.hyperparameters = hyperparameters
        self.batch_size_per_gp_sample = batch_size_per_gp_sample
