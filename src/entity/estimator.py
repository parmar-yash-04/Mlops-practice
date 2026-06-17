from src.constants import TARGET_COLUMN


class TargetValueMapping:
    def __init__(self):
        self.mapping = {"Y": 1, "N": 0}

    def get_mapping(self) -> dict:
        return self.mapping

    def inverse_mapping(self) -> dict:
        return {v: k for k, v in self.mapping.items()}

    def transform(self, value):
        return self.mapping.get(value)

    def __len__(self):
        return len(self.mapping)
