import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import get_linear_schedule_with_warmup


class EncodersLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, encoded_q, encoded_c):
        scores = encoded_q.mm(encoded_c.t())
        target = torch.arange(scores.shape[0]).to(scores.device)
        loss = F.cross_entropy(scores, target)
        return loss


class BiEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loss_history = []
        self.lr_history = []
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
        self.c_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
        self.q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")    
        self.c_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
        self.criterion = EncodersLoss()     
        self.to(self.device)
    
    def forward(self, question, context):
        encoded_q = self.encode(question, self.q_encoder, self.q_tokenizer)
        encoded_c = self.encode(context, self.c_encoder, self.c_tokenizer)
        return encoded_q, encoded_c
    
    def encode(self, text, encoder, tokenizer):
        tokenized = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors="pt")
        tokenized = tokenized.to(self.device)
        encoded = encoder(**tokenized).pooler_output
        return encoded
    
    def train_(self, data_loader, nb_epochs, batch_size, lr=1e-5, verbose=True):
        grad_acc_steps = int(128/batch_size)    
        nb_batch_steps = nb_epochs*int(len(data_loader)/grad_acc_steps)
        nb_warmup_steps = int(0.01*nb_batch_steps)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=nb_warmup_steps, num_training_steps=nb_batch_steps)
        
        self.train()
        for e in range(nb_epochs):
            train_loss = 0
            batch_step = 0
            
            start = time.time()
            for train_step, (questions, contexts) in enumerate(data_loader):
                ################################################################################
                # Training process                                                             #
                ################################################################################
                encoded_q, encoded_c = self(questions, contexts)        # Forward pass (computational graph generation)
                loss = self.criterion(encoded_q, encoded_c)             # Loss computation
                loss.backward()                                         # Backward pass (gradient computation over graph)
                train_loss += loss.item() 
                
                # Gradient accumulation that simulate the given batch size
                if (train_step+1) % grad_acc_steps == 0:
                    optimizer.step()                                    # Weight updates
                    scheduler.step()                                    # Learning rate adjustement
                    optimizer.zero_grad()
                    self.train_loss_history.append(train_loss/grad_acc_steps)
                    train_loss = 0
                    batch_step += 1
                    batch_time = time.time() - start
                    start = time.time()
                    
                    if verbose:
                        print("Batch step: {}/{}, Train loss: {:.4f}, Time: {:.2f}s/batch".format(batch_step, 
                                                                                            nb_batch_steps, 
                                                                                            self.train_loss_history[-1],
                                                                                            batch_time))
                
                for param_group in optimizer.param_groups:
                    self.lr_history.append(param_group['lr'])
    
    def load_pretrained_model(self, model_path):
        m_state_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.load_state_dict(m_state_dict)