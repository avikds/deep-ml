import torch
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model


def train(model, tokenizer, train_texts, val_texts):
    # GPT-2 tokenizers often have no pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx]

    def collate(batch):
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        # Causal LM: labels are the input tokens.
        labels = enc["input_ids"].clone()

        # Don't calculate loss on padding tokens.
        labels[enc["attention_mask"] == 0] = -100

        enc["labels"] = labels
        return enc

    # ---------------------------------------------------------
    # LoRA
    # ---------------------------------------------------------
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    device = next(model.parameters()).device
    model.train()

    loader = DataLoader(
        TextDataset(train_texts),
        batch_size=4,
        shuffle=True,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=5e-4,
        weight_decay=0.01,
    )

    # Keep training within the lab's time budget.
    epochs = 3

    for epoch in range(epochs):
        model.train()

        for batch in loader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                outputs = model(**batch)
                loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

    model.eval()
    return model