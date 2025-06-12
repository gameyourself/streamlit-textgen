# -*- coding: utf-8 -*-
"""
Created on Sun May 18 17:32:04 2025

@author: 86136
"""

# -*- coding: utf-8 -*-
import numpy as np
import jieba
import re
import json
from gensim.models import Word2Vec
from keras.models import Sequential, save_model
from keras.layers import LSTM, Dense, Embedding

def preprocess_text(text, is_poetry=True):
    """文本预处理（支持古诗/现代文）"""
    text = re.sub(r'[^\u4e00-\u9fa5，。！？、：；《》【】（）“”‘’…—]', '', text)
    return list(text) if is_poetry else jieba.lcut(text)

def create_sequences(data, seq_length=20, step=3):
    """创建训练序列"""
    sequences = []
    next_items = []
    for i in range(0, len(data)-seq_length, step):
        sequences.append(data[i:i+seq_length])
        next_items.append(data[i+seq_length])
    return sequences, next_items

def main():
    # 配置参数
    config = {
        'seq_length': 20,
        'epochs': 30,
        'batch_size': 64,
        'lstm_units': 256,
        'is_poetry': True,
        'train_data_path': 'poetry.txt',
        'model_save_path': 'models/poetry_model.h5',
        'vocab_save_path': 'models/vocab.json',
        'word2vec_save_path': 'models/word2vec.model'
    }

    # 数据加载与预处理
    with open(config['train_data_path'], 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    data = preprocess_text(text, config['is_poetry'])
    
    # 创建词汇表
    vocab = sorted(set(data))
    word2idx = {w:i for i,w in enumerate(vocab)}
    idx2word = {i:w for i,w in enumerate(vocab)}
    
    # 生成训练数据
    sequences, next_words = create_sequences(data, config['seq_length'])
    
    # 向量化处理
    X = np.zeros((len(sequences), config['seq_length'], len(vocab)), dtype=np.bool_)
    y = np.zeros((len(sequences), len(vocab)), dtype=np.bool_)
    for i, seq in enumerate(sequences):
        for t, word in enumerate(seq):
            X[i,t,word2idx[word]] = 1
        y[i,word2idx[next_words[i]]] = 1

    # 训练Word2Vec
    w2v_model = Word2Vec(sentences=[data], vector_size=100, window=5, min_count=1)
    w2v_model.save(config['word2vec_save_path'])

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(config['lstm_units'], return_sequences=True,
                input_shape=(config['seq_length'], len(vocab))))
    model.add(LSTM(config['lstm_units']//2))
    model.add(Dense(len(vocab), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # 保存模型和配置
    model.save(config['model_save_path'])
    with open(config['vocab_save_path'], 'w', encoding='utf-8') as f:
        json.dump({
            'word2idx': {str(k): v for k, v in word2idx.items()},  # 转换为字符串键
            'idx2word': {str(k): v for k, v in idx2word.items()},
            'is_poetry': config['is_poetry'],
            'seq_length': config['seq_length'],
            'vocab_size': len(vocab)
        }, f, ensure_ascii=False)

if __name__ == "__main__":
    main()