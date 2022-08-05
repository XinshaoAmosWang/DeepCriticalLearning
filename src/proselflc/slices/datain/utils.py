import numpy as np
from torch.utils.data.sampler import BatchSampler


def set_torch_seed(random=None, numpy=None, torch=None, os=None, seed=0):
    if random is not None:
        random.seed(seed)
    if numpy is not None:
        numpy.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
    if os is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)


# From https://github.com/adambielski/
# siamese-triplet/blob/master/datasets.py
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler -
        from a MNIST-like dataset,
        samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples, seed=0):
        # It is fine to always force reproducibility
        # for numpy and batch sampler, without input dependence.
        np.random.seed(seed)

        self.labels = np.array(labels).astype(np.int64)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {
            label: np.where(self.labels == label)[0] for label in self.labels_set
        }
        for _label in self.labels_set:
            np.random.shuffle(self.label_to_indices[_label])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                        range(
                            self.used_label_indices_count[class_],
                            self.used_label_indices_count[class_] + self.n_samples,
                        )
                    ]
                )
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(
                    self.label_to_indices[class_]
                ):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size
