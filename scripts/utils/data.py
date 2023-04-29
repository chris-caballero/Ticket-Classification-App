from torch.utils.data import Dataset, DataLoader

class TicketDataset(Dataset):
    def __init__(self, data, tokenizer, field='complaint_nouns', block_size=200):
        self.data = data
        self.field = field
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[self.field].iloc[idx]
        label = self.data['label'].iloc[idx]

        # Tokenize the text and create input_ids and attention_mask tensors
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.block_size,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }


def create_dataloader(dataset, batch_size=16):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True
    )


def to_dataloader(dataset, batch_size=16, split=0.8):
    from torch.utils.data import random_split

    if split >= 1 or split <= 0:
        return create_dataloader(dataset, batch_size=batch_size)

    # Define the length of the dataset and the sizes of the training and testing sets
    train_size = int(len(dataset) * split)
    test_size = len(dataset) - train_size

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders for the training and testing sets
    train = create_dataloader(train_dataset, batch_size=batch_size)
    test = create_dataloader(test_dataset, batch_size=batch_size)

    return train, test
