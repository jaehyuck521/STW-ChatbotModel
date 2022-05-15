

import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F 
import pickle




# Setup seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


cuda_num = "cuda:0"

class PositionalEncoder(nn.Module):
   
    def __init__(self, position, d_model):
        super().__init__()

        self.d_model = d_model  # original voca dimensions

        # index
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
        # x+ positional encoding
        ret = math.sqrt(self.d_model)*x + self.pe[:, :x.size(1)]
        return ret


# In[ ]:


def scaled_dot_product_attention(query, key, value, mask):
  # query size : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
  # key size : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
  # value size : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
  # padding_mask : (batch_size, 1, 1, key의 문장 길이)

  # attention score, matrixx
  matmul_qk = torch.matmul(query, torch.transpose(key,2,3))

  # scaling
  depth = key.shape[-1]
  logits = matmul_qk / math.sqrt(depth)

  # masking
  if mask is not None:
    logits += (mask * -1e9)

  
  # attention weight : (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
  attention_weights = F.softmax(logits, dim=-1)

  # output : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
  output = torch.matmul(attention_weights, value)

  return output, attention_weights


# In[ ]:


class MultiheadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0
        
        self.depth = int(d_model/self.num_heads)

        # WQ, WK, WV
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        # WO
        self.out = nn.Linear(d_model, d_model)


    # num_heads split function
    def split_heads(self, inputs, batch_size):
      inputs = torch.reshape(
          inputs, (batch_size, -1, self.num_heads, self.depth))
      return torch.transpose(inputs, 1,2)

    def forward(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = query.shape[0]
        # 1. WQ, WK, WV
        # q : (batch_size, query의 문장 길이, d_model)
        # k : (batch_size, key의 문장 길이, d_model)
        # v : (batch_size, value의 문장 길이, d_model)
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)


        # 2. divide head
        # q : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # k : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        # v : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)


        # 3. scale dot product
        # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
        scaled_attention = torch.transpose(scaled_attention, 1,2)

        # 4. connect head
        # (batch_size, query의 문장 길이, d_model)
        concat_attention = torch.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # 5. WO layer pass
        # (batch_size, query의 문장 길이, d_model)
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




def create_padding_mask(x):
  input_pad = 0
  mask = (x == input_pad).float()
  mask = mask.unsqueeze(1).unsqueeze(1)
  # (batch_size, 1, 1, key의 문장 길이)
  return mask


# In[ ]:


def create_look_ahead_mask(x):
  device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")
  seq_len = x.shape[1]
  look_ahead_mask = torch.ones(seq_len, seq_len)
  look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1).to(device)

  padding_mask = create_padding_mask(x).to(device) # 패딩 마스크도 포함
  return torch.maximum(look_ahead_mask, padding_mask)


# # Train Transformer model



import pandas as pd
import os

from soynlp.tokenizer import LTokenizer

# chatbot using code, get the model and use them 

def get_chatbot_instance(model_name, output_dir, num_epoch):
    ## Load pickle
    with open(os.path.join(output_dir, "transformermodel_" + model_name + "_Q.pickle"),"rb") as fr:
        Q = pickle.load(fr)
        
    with open(os.path.join(output_dir, "transformermodel_" + model_name + "_A.pickle"),"rb") as fr:
        A = pickle.load(fr)
        
    with open(os.path.join(output_dir, "transformermodel_" + model_name + "_HYPE.pickle"),"rb") as fr:
        [VOCAB_SIZE, text_embedding_vectors, NUM_LAYERS, D_FF, D_MODEL, NUM_HEADS, DROPOUT] = pickle.load(fr)
    VOCAL_SIZE_ = VOCAB_SIZE
    PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN = Q.vocab.stoi['<pad>'], Q.vocab.stoi['<SOS>'], Q.vocab.stoi['<EOS>'], Q.vocab.stoi['<unk>']
    cuda_num = "cuda:0"
    device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")
            
    net_trained = transformer(text_embedding_vectors = text_embedding_vectors, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, d_ff=D_FF, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT).to(device)
    net_trained.load_state_dict(torch.load(os.path.join(output_dir, "transformermodel_" + model_name + "_model_"+str(num_epoch)+".pt")))
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
    

    
    def predict(sentence):
      out = evaluate(net_trained, sentence)
      out_text = itos(Q.vocab.itos, out)
      print('input = [{0}]'.format(sentence))
      print('output = [{0}]'.format(out_text))
      return out_text
  
    
  
    return predict






model_name = "v3_preprocessed"
output_dir = "./snapshot_v3_preprocessed_DModel256_batch64_vocab400_minfreq25_preprocessed2_data"
num_epoch = 99



predict_instance = get_chatbot_instance(model_name, output_dir, num_epoch)


print(predict_instance("심심해"))
print(predict_instance("하고 싶은 말이 있어"))
print(predict_instance("제대로 작동하는거 맞아?"))
print(predict_instance("왜 이렇게 오래 걸래"))





while True:
    print("answer : ", predict_instance(input("type question : ")))