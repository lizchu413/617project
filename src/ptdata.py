from torch.utils.data import Dataset
import pandas as pd


class CodeDataset(Dataset):
    def __init__(self, bodies: list, descriptions: list, declarations: list) -> None:
        self.data = pd.DataFrame(
            list(zip(bodies, descriptions, declarations)),
            columns=["body", "desc", "decl"],
        )
        self.data["length"] = self.data["body"].map(
            lambda x: 1 + (x.split(" ").count("DCNL"))
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> dict:
        row = self.data.iloc[index]
        return row
