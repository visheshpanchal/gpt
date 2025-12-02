import torch

from config import config, Config
from data_loader import prepare_data
from model import GPT


class Trainer:

    def __init__(self, _config: Config):
        self.model = GPT(_config)
        self._config = _config
        self.optim = torch.optim.AdamW(self.model.parameters(),
                                          lr=_config.learning_rate, weight_decay=0.1)


    def datasets(self):
        data = prepare_data(
            self._config.data.data_dir,
            self._config.data.data_name,
            self._config.data.data_dir,
            'train',
            self._config.data.model_name_from_hf,
            max_length= self._config.data.max_length,
            stride= self._config.data.stride,
            max_tokens= self._config.data.max_tokens
        )

        return data


    def checkpoint(self):
        pass

    def calculate_loss(self, input_tensor, target_tensor):
        input_batch, target_batch = input_tensor.to(self._config.device), target_tensor.to(self._config.device)
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    def eval_model(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            train_loss = self.calculate_loss(input_tensor, target_tensor)
            val_loss = self.calculate_loss(input_tensor, target_tensor)

        self.model.train()
        return train_loss, val_loss


    def train(self):
        records = self.datasets()

        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen = 0
        global_step = -1

        for epoch in range(config.no_of_epoch):
            for record in records:

                # Reset Gradient to Zero
                self.optim.zero_grad()

                input_tensor, target_tensor = record

                loss = self.calculate_loss(input_tensor, target_tensor)

                # Calculating loss gradients
                loss.backward()

                # Update model weights using loss gradients
                self.optim.step()

                tokens_seen += input_tensor.numel()
                global_step += 1

                # Eval loop
                if global_step % 10 == 0:
                    train_loss, val_loss = self.eval_model(input_tensor, target_tensor)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        return train_losses, val_losses, track_tokens_seen






if __name__ == '__main__':
    trainer = Trainer(config)
    trainer.train()
