#!/usr/bin/env python


# In[ ]:


import math
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchtext
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# In[ ]:


# Setup seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# # transformer model implementation

# In[ ]:
cuda_num = "cuda:0"
model_name = "v3_preprocessed"
output_dir = "./snapshot_" + model_name

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
    f = open(os.path.join(output_dir, "transformermodel_" + model_name + "_loss.log"), "w")
    f.write("")
    f.close()
    f = open(os.path.join(output_dir, "transformermodel_" + model_name + "_QnA.log"), "w")
    f.write("")
    f.close()
else:
    print("warning!!! there is the folder")
    input()
    
    # Positional encoding
class PositionalEncoder(nn.Module):
    
    def __init__(self, position, d_model):
        super().__init__()

        self.d_model = d_model  

       
        pe = torch.zeros(position, d_model)

       
        device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)

        for pos in range(position):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * i)/d_model)))

        
        self.pe = pe.unsqueeze(0)

        self.pe.requires_grad = False

    def forward(self, x):
       
        ret = math.sqrt(self.d_model)*x + self.pe[:, :x.size(1)]
        return ret


# In[ ]:


def scaled_dot_product_attention(query, key, value, mask):
 


  matmul_qk = torch.matmul(query, torch.transpose(key,2,3))

  
  depth = key.shape[-1]
  logits = matmul_qk / math.sqrt(depth)


  if mask is not None:
    logits += (mask * -1e9)

  
  attention_weights = F.softmax(logits, dim=-1)

  
  output = torch.matmul(attention_weights, value)

  return output, attention_weights


# In[ ]:

# multi head attention apply
class MultiheadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0
        
        self.depth = int(d_model/self.num_heads)

        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.out = nn.Linear(d_model, d_model)


    
    def split_heads(self, inputs, batch_size):
      inputs = torch.reshape(
          inputs, (batch_size, -1, self.num_heads, self.depth))
      return torch.transpose(inputs, 1,2)

    def forward(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = query.shape[0]
        
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)


      
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)


        
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        
        scaled_attention = torch.transpose(scaled_attention, 1,2)

        
        concat_attention = torch.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        outputs = self.out(concat_attention)
        return outputs


# In[ ]:


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, attention):
        outputs = self.linear_1(attention)
        outputs = F.relu(outputs)
        outputs = self.linear_2(outputs)
        return outputs


# In[ ]:


class EncoderBlock(nn.Module):
  def __init__(self, d_ff, d_model, num_heads, dropout):
    super(EncoderBlock, self).__init__()
    
    self.attn = MultiheadAttention(d_model, num_heads)
    self.dropout_1 = nn.Dropout(dropout)
    self.norm_1 = nn.LayerNorm(d_model)
    self.ff = FeedForward(d_model, d_ff)
    self.dropout_2 = nn.Dropout(dropout)
    self.norm_2 = nn.LayerNorm(d_model)

  def forward(self, inputs, padding_mask):
    attention = self.attn({'query': inputs, 'key': inputs, 'value': inputs, 'mask': padding_mask})
    attention = self.dropout_1(attention)
    attention = self.norm_1(inputs + attention)
    outputs = self.ff(attention)
    outputs = self.dropout_2(outputs)
    outputs = self.norm_2(attention + outputs)

    return outputs


# In[ ]:


class Encoder(nn.Module):
  def __init__(self,text_embedding_vectors, vocab_size, num_layers, d_ff, d_model, num_heads, dropout):
    super(Encoder, self).__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.num_layers = num_layers
    self.embb = nn.Embedding(text_embedding_vectors, d_model)
    self.dropout_1 = nn.Dropout(dropout)
    self.PE = PositionalEncoder(vocab_size, d_model)
    self.encoder_block = EncoderBlock(d_ff, d_model, num_heads, dropout)
  def forward(self, x, padding_mask):
    emb = self.embb(x)
    emb *= math.sqrt(self.d_model)
    emb = self.PE(emb)
    output = self.dropout_1(emb)

    for i in range(self.num_layers):
      output = self.encoder_block(output, padding_mask)

    return output


# In[ ]:


class DecoderBlock(nn.Module):
  def __init__(self, d_ff, d_model, num_heads, dropout):
    super(DecoderBlock, self).__init__()
    
    self.attn = MultiheadAttention(d_model, num_heads)
    self.attn_2 = MultiheadAttention(d_model, num_heads)
    self.dropout_1 = nn.Dropout(dropout)
    self.norm_1 = nn.LayerNorm(d_model)
    self.ff = FeedForward(d_model, d_ff)
    self.dropout_2 = nn.Dropout(dropout)
    self.dropout_3 = nn.Dropout(dropout)
    self.norm_2 = nn.LayerNorm(d_model)
    self.norm_3 = nn.LayerNorm(d_model)

  def forward(self, inputs, enc_outputs, padding_mask, look_ahead_mask):
    attention1 = self.attn({'query': inputs, 'key': inputs, 'value': inputs, 'mask': look_ahead_mask})
    attention1 = self.norm_1(inputs + attention1)
    attention2 = self.attn_2({'query': attention1, 'key': enc_outputs, 'value': enc_outputs, 'mask': padding_mask})
    attention2 = self.dropout_1(attention2)
    attention2 = self.norm_2(attention1 + attention2)

    outputs = self.ff(attention2)
    outputs = self.dropout_3(outputs)
    outputs = self.norm_3(attention2 + outputs)

    return outputs  


# In[ ]:


class Decoder(nn.Module):
  def __init__(self,text_embedding_vectors,  vocab_size, num_layers, d_ff, d_model, num_heads, dropout):
    super(Decoder, self).__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.num_layers = num_layers
    self.embb = nn.Embedding(text_embedding_vectors, d_model)
    self.dropout_1 = nn.Dropout(dropout)
    self.PE = PositionalEncoder(vocab_size, d_model)
    self.decoder_block = DecoderBlock(d_ff, d_model, num_heads, dropout)
  def forward(self, enc_output, dec_input, padding_mask, look_ahead_mask):
    emb = self.embb(dec_input)
    emb *= math.sqrt(self.d_model)
    emb = self.PE(emb)
    output = self.dropout_1(emb)
    for i in range(self.num_layers):
      output = self.decoder_block(output, enc_output, padding_mask, look_ahead_mask)

    return output


# In[ ]:


class transformer(nn.Module):
    def __init__(self, text_embedding_vectors, vocab_size, num_layers, d_ff, d_model, num_heads, dropout):
        self.vocab_size = vocab_size
        super(transformer, self).__init__()
        self.enc_outputs = Encoder(text_embedding_vectors, vocab_size, num_layers, d_ff, d_model, num_heads, dropout)
        self.dec_outputs = Decoder(text_embedding_vectors, vocab_size, num_layers, d_ff, d_model, num_heads, dropout)
        self.output = nn.Linear(d_model, text_embedding_vectors)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, dec_input):
        enc_input = input
        dec_input = dec_input
        enc_padding_mask = create_padding_mask(enc_input)
        dec_padding_mask = create_padding_mask(enc_input)
        look_ahead_mask = create_look_ahead_mask(dec_input)
    
        enc_output = self.enc_outputs(enc_input, enc_padding_mask)
        dec_output = self.dec_outputs(enc_output, dec_input, dec_padding_mask, look_ahead_mask)
        output = self.output(dec_output)
        return output


# In[ ]:


import pandas as pd
import re
import urllib.request
import time




# In[ ]:



import os
file_list = ['preprocessed3_data.csv']#['chatbot_preprocessed.csv']

train_data = pd.read_csv(file_list[0])
train_data.head()


# In[ ]:


from torchtext.legacy import data, datasets
import os


# # soynlp tokenizer

# In[ ]:


#https://github.com/lovit/soynlp
#get_ipython().system('pip install soynlp')


# In[ ]:


from soynlp.tokenizer import LTokenizer


# In[ ]:


tokenizer = LTokenizer()


# In[ ]:


tokenizer("내일 역 앞의 식당에서 밥 먹으러 나갈래 ?")


# In[ ]:


VOCAL_SIZE_ = 400#60,250


# # set the data field , question and answer

# In[ ]:


Q = data.Field(
    sequential=True,
    use_vocab=True,
    lower=True,
    tokenize=tokenizer,
    batch_first=True,
    init_token="<SOS>",
    eos_token="<EOS>",
    fix_length=VOCAL_SIZE_
)

A = data.Field(
    sequential=True,
    use_vocab=True,
    lower=True,
    tokenize=tokenizer,
    batch_first=True,
    init_token="<SOS>",
    eos_token="<EOS>",
    fix_length=VOCAL_SIZE_
)


# # set the dataset

# In[ ]:


trainset = data.TabularDataset(
        path=file_list[0], format='csv', skip_header=False,
        fields=[('Q', Q),('A', A)])


# In[ ]:


print(vars(trainset[2]))


# In[ ]:


print('훈련 샘플의 개수 : {}'.format(len(trainset)))


# In[ ]:


Q.build_vocab(trainset.Q, trainset.A, min_freq = 25) 
A.vocab = Q.vocab


# In[ ]:


PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN = Q.vocab.stoi['<pad>'], Q.vocab.stoi['<SOS>'], Q.vocab.stoi['<EOS>'], Q.vocab.stoi['<unk>']


# In[ ]:


#Difine HyperParameter
VOCAB_SIZE = VOCAL_SIZE_
text_embedding_vectors = len(Q.vocab)
NUM_LAYERS = 4
D_FF = 512
D_MODEL = 256#128,256
NUM_HEADS = 4
DROPOUT = 0.3
BATCH_SIZE=64#1024,100


# In[ ]:


# Define Iterator
# train_iter batch has text and target item
device = torch.device(cuda_num if torch.cuda.is_available() else 'cpu')

train_iter = data.BucketIterator(
        trainset, batch_size=BATCH_SIZE,
        shuffle=True, repeat=False, sort=False, device = device)


# In[ ]:


# model build
print(text_embedding_vectors)
net = transformer(text_embedding_vectors = text_embedding_vectors, 
                  vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, d_ff=D_FF, d_model=D_MODEL, 
                  num_heads=NUM_HEADS, dropout=DROPOUT)

# initialize network
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
       
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


net.train()


net.apply(weights_init)


print('네트워크 초기화 완료')


# In[ ]:


# loss-function
criterion = nn.CrossEntropyLoss()


learning_rate = 2e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# In[ ]:


def create_padding_mask(x):
  input_pad = 0
  mask = (x == input_pad).float()
  mask = mask.unsqueeze(1).unsqueeze(1)
 
  return mask


# In[ ]:


def create_look_ahead_mask(x):
  device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")
  seq_len = x.shape[1]
  look_ahead_mask = torch.ones(seq_len, seq_len)
  look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1).to(device)

  padding_mask = create_padding_mask(x).to(device) 
  return torch.maximum(look_ahead_mask, padding_mask)


# # Train Transformer model

# In[ ]:


def stoi(vocab, token, max_len):
  #
  indices=[]
  token.extend(['<pad>'] * (max_len - len(token)))
  for string in token:
    if string in vocab:
      i = vocab.index(string)
    else:
      i = 0
    indices.append(i)
  return torch.LongTensor(indices).unsqueeze(0)

def itos(vocab, indices):
  text = []
  for i in indices.cpu()[0]:
    if i==1:
      break
    else:
      if i not in [PAD_TOKEN, START_TOKEN, END_TOKEN]:
          if i != UNK_TOKEN:
              text.append(vocab[i])
          else:
              text.append('??')
  return " ".join(text)


# In[ ]:


def evaluate(net_trained, input_sentence):
    VOCAB_SIZE = VOCAL_SIZE_
    device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")
    tokenizer = LTokenizer()
    token = tokenizer(input_sentence)
    input = stoi(Q.vocab.itos, token, VOCAB_SIZE).to(device)
    output = torch.LongTensor(1, 1).fill_(START_TOKEN).to(device)
    for i in range(VOCAB_SIZE):
        predictions = net_trained(input, output)
        predictions = predictions[:, -1:, :]
                            
        #                                      PAD, UNK, START 토큰 제외
        predicted_id = torch.argmax(predictions[:,:,3:], axis=-1) + 3
        if predicted_id == END_TOKEN:
            predicted_id = predicted_id
            break
        output = torch.cat((output, predicted_id),-1)
    return output


# In[ ]:


def predict(net_trained, sentence):
  out = evaluate(net_trained, sentence)
  out_text = itos(Q.vocab.itos, out)
  print('input = [{0}]'.format(sentence))
  print('output = [{0}]'.format(out_text))
  return out_text




from torch.nn.parallel import DistributedDataParallel as DDP

from IPython.display import clear_output
import datetime

def train_model(net, train_iter, criterion, optimizer, num_epochs):
    start_time = time.time()

    ntokens = len(Q.vocab.stoi)
    device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")
    print("사용 디바이스:", device)
    print('-----start-------')
    net.to(device)
    epoch_ = []
    epoch_train_loss = []
   
    torch.backends.cudnn.benchmark = True
    
    net.train()
    
    best_epoch_loss = float("inf")
    for epoch in range(num_epochs):
      epoch_loss = 0.0
      cnt= 0
      for batch in train_iter:
          questions = batch.Q.to(device)
          answers = batch.A.to(device)
          with torch.set_grad_enabled(True):
            
            preds = net(questions, answers)
            pad = torch.LongTensor(answers.size(0), 1).fill_(PAD_TOKEN).to(device)
            preds_id = torch.transpose(preds,1,2)
            outputs = torch.cat((answers[:, 1:], pad), -1)
            optimizer.zero_grad()
            loss = criterion(preds_id, outputs)  # loss 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            epoch_loss +=loss.item()
            cnt += 1
      epoch_loss = epoch_loss / cnt
      torch.save(net.state_dict(), os.path.join(output_dir, "transformermodel_" + model_name + "_model_"+str(epoch)+".pt"))
      f = open(os.path.join(output_dir, "transformermodel_" + model_name + "_loss.log"), "a")
      f.write(",".join([str(epoch), str(epoch_loss)]) + "\n")
      f.close()
      
      epoch_.append(epoch)
      epoch_train_loss.append(epoch_loss)
      print('Epoch {0}/{1} Average Loss: {2}'.format(epoch+1, num_epochs, epoch_loss))
      
      qs = ['우리 내일 같이 영화 볼래?', '그 영화 너무 별로더라', '제대로 작동하는거 맞아?', '내일 뭐 먹고 싶어?', '오는데 몇시간 걸려?']
      res = [predict(net, x) for x in qs]
      qna = [str(epoch)]
      for x in range(len(qs)):
          qna.append(qs[x])
          qna.append(res[x])
      
      f = open(os.path.join(output_dir, "transformermodel_" + model_name + "_QnA.log"), "a")
      f.write(",".join(qna) + "\n")
      f.close()
      
      
      clear_output(wait = True)
    
    
    fig = plt.figure(figsize=(8,8))
    fig.set_facecolor('white')
    ax = fig.add_subplot()

    ax.plot(epoch_,epoch_train_loss, label='Average loss')


    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')

    plt.show()
    end_time = time.time() - start_time
    times = str(datetime.timedelta(seconds=end_time)).split(".")
    print('Finished in {0}'.format(times[0]))


# In[ ]:


num_epochs = 100#30
train_model(net, train_iter, criterion, optimizer, num_epochs=num_epochs)


# In[ ]:


net_trained = transformer(text_embedding_vectors = text_embedding_vectors, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, d_ff=D_FF, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT).to(device)
net_trained.load_state_dict(torch.load(os.path.join(output_dir, "transformermodel_" + model_name + "_model_"+str(num_epochs-1)+".pt")))


# # Run Transformer chatbot for real sentence

# In[ ]:





out = predict(net_trained, '우리 내일 같이 영화 볼래?')
out = predict(net_trained, '그 영화 너무 별로더라')
out = predict(net_trained, '제대로 작동하는거 맞아?')
out = predict(net_trained, '내일 뭐 먹고 싶어?')
out = predict(net_trained, '오는데 몇시간 걸려?')


import pickle
with open(os.path.join(output_dir, "transformermodel_" + model_name + "_Q.pickle"),"wb") as fw:
    pickle.dump(Q, fw)

with open(os.path.join(output_dir, "transformermodel_" + model_name + "_A.pickle"),"wb") as fw:
    pickle.dump(Q, fw)

with open(os.path.join(output_dir, "transformermodel_" + model_name + "_HYPE.pickle"),"wb") as fw:
    pickle.dump([VOCAB_SIZE, text_embedding_vectors, NUM_LAYERS, D_FF, D_MODEL, NUM_HEADS, DROPOUT], fw)
    

