from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, epochs, train_loader, valid_loader, model, optimizer,
                 device=None, scheduler=None, early_stopping=10, save_dir='./checkpoints',
                 load_best_model=False, grad_clip=None, is_lora=False):
        self.epochs = epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.load_best_model = load_best_model
        self.grad_clip = grad_clip
        self.is_lora = is_lora

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print('Using device:', self.device)
        else:
            self.device = device

        self.model = model.to(self.device)

        self.save_dir = save_dir
        self.save_name = 'best_model.ckpt'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def train_epoch(self, epoch):
        train_loss = 0
        train_pbar = tqdm(self.train_loader, position=0, leave=True)
        self.model.train()

        for input_datas in train_pbar:
            self.optimizer.zero_grad()
            input_datas = {k: v.to(self.device) for k, v in input_datas.items()}
            outputs = self.model(**input_datas)
            loss = outputs[0]
            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            train_pbar.set_description(f'Train Epoch {epoch}')
            train_pbar.set_postfix({'loss': f'{loss.item():.3f}'})

            train_loss += loss.item()

        return train_loss / len(self.train_loader)

    def validate_epoch(self, epoch):
        valid_loss = 0
        valid_pbar = tqdm(self.valid_loader, position=0, leave=True)
        self.model.eval()

        with torch.no_grad():
            for input_datas in valid_pbar:
                input_datas = {k: v.to(self.device) for k, v in input_datas.items()}
                outputs = self.model(**input_datas)
                loss = outputs[0]
                valid_pbar.set_description(f'Valid Epoch {epoch}')
                valid_pbar.set_postfix({'loss': f'{loss.item():.3f}'})
                valid_loss += loss.item()

        return valid_loss / len(self.valid_loader)

    def train(self, show_loss=True):
        best_loss = float('inf')
        loss_record = {'train': [], 'valid': []}
        stop_cnt = 0

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            valid_loss = self.validate_epoch(epoch)

            loss_record['train'].append(train_loss)
            loss_record['valid'].append(valid_loss)

            # Save best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                if self.is_lora:
                    self.model.save_pretrained(self.save_dir)
                else:
                    save_path = os.path.join(self.save_dir, self.save_name)
                    torch.save(self.model.state_dict(), save_path)
                print(f'Saving Model With Loss {best_loss:.5f}')
                stop_cnt = 0
            else:
                stop_cnt += 1

            print(f'Train Loss: {train_loss:.5f} | Valid Loss: {valid_loss:.5f} | Best Loss: {best_loss:.5f}\n')

            if stop_cnt == self.early_stopping:
                msg = "Model can't improve, stop training"
                print('-' * (len(msg) + 4))
                print(f'| {msg} |')
                print('-' * (len(msg) + 4))
                break

        if show_loss:
            self.show_training_loss(loss_record)

        if self.load_best_model:
            if self.is_lora:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, self.save_dir)
                print(f'Best LoRA model loaded from {self.save_dir}')
            else:
                best_model_path = os.path.join(self.save_dir, self.save_name)
                self.model.load_state_dict(torch.load(best_model_path))
                print(f'Best model loaded from {best_model_path}')

    def show_training_loss(self, loss_record):
        train_loss, valid_loss = [i for i in loss_record.values()]
        plt.plot(train_loss)
        plt.plot(valid_loss)
        plt.title('Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.show()
