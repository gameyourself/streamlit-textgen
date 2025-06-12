# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:02:03 2025

@author: 86136
"""

# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import json
import jieba
from gensim.models import Word2Vec
from keras.models import load_model

def load_resources():
    """加载模型和配置"""
    try:
        model = load_model('C:/Users/xiexiangfei/Desktop/Uibe-Learning/文本挖掘大作业/HW4文本生成/models/poetry_model.h5')
        w2v_model = Word2Vec.load('C:/Users/xiexiangfei/Desktop/Uibe-Learning/文本挖掘大作业/HW4文本生成/models/word2vec.model')
        with open('C:/Users/xiexiangfei/Desktop/Uibe-Learning/文本挖掘大作业/HW4文本生成/models/vocab.json', 'r',encoding='utf-8') as f:
            vocab = json.load(f)
        return model, w2v_model, vocab
    except Exception as e:
        st.error(f"加载失败: {str(e)}")
        return None, None, None

def main():
    st.set_page_config(page_title="文本生成实验室", layout="wide")
    
    # 侧边栏设置
    with st.sidebar:
        st.header("控制面板")
        uploaded_file = st.file_uploader("上传语料文件", type=["txt"])
        
        # 文件预览
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8", errors='ignore')
            preview_len = st.slider("预览字数", 100, 500, 200)
            with st.expander("文件预览"):
                st.text_area("内容", text[:preview_len], height=200)
    
    # 主界面功能
    tabs = st.tabs(["文本分割", "词向量", "模型代码", "文本生成"])
    model, w2v_model, vocab = load_resources()
    
    # 文本分割展示
    with tabs[0]:
        st.header("文本分割示例")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("字符分割")
            sample_text = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
            window_size = st.slider("字符窗口大小", 3, 10, 5, key="char_win")
            char_sequences = [sample_text[i:i+window_size] 
                             for i in range(0, len(sample_text)-window_size, 3)]
            st.code(f"输入序列: {char_sequences[0]}\n预测字符: {sample_text[window_size]}",
                   language="python")
        
        with col2:
            st.subheader("词级分割")
            sample_text = "人工智能是未来科技发展的重要方向"
            words = jieba.lcut(sample_text)
            window_size = st.slider("词窗口大小", 2, 5, 3, key="word_win")
            word_sequences = [words[i:i+window_size] 
                             for i in range(0, len(words)-window_size, 1)]
            st.code(f"输入序列: {word_sequences[0]}\n预测词: {words[window_size]}",
                   language="python")
    
    # 词向量展示
    with tabs[1]:
        if w2v_model:
            st.header("词向量信息")
            st.write(f"向量维度: {w2v_model.vector_size}")
            st.write(f"词汇表大小: {len(w2v_model.wv)}")
            sample_word = st.selectbox("示例词查看", list(w2v_model.wv.index_to_key)[:50])
            st.code(f"'{sample_word}' 向量示例:\n{w2v_model.wv[sample_word]}", 
                   language="python")
    
    # 模型代码展示
    with tabs[2]:
        st.header("模型架构代码")
        with open('C:/Users/xiexiangfei/Desktop/Uibe-Learning/文本挖掘大作业/HW4文本生成/train.py', 'r', encoding='utf-8', errors='ignore') as f:
            model_code = f.read()
        st.code(model_code, language='python')
    
    # 文本生成功能
    with tabs[3]:
        st.header("文本生成器")
        if model and vocab:
            col1, col2 = st.columns([1, 3])
            with col1:
                with st.form("generate_form"):
                    seed_text = st.text_area("输入起始文本", "床前明月光", height=100)
                    gen_length = st.slider("生成长度", 50, 500, 100)
                    temperature = st.slider("创意度", 0.1, 2.0, 1.0, 0.1)
                    if st.form_submit_button("生成文本"):
                        with st.spinner("生成中..."):
                            generated = generate_text(seed_text, gen_length, 
                                                    temperature, model, vocab)
                            st.session_state.generated = generated
            with col2:
                if 'generated' in st.session_state:
                    st.text_area("生成结果", st.session_state.generated, height=300)
def sample_with_temp(preds, temp):
    # 添加数值稳定性处理
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temp  # 防止log(0)
    exp_preds = np.exp(preds - np.max(preds))  # 数值稳定处理
    probs = exp_preds / (np.sum(exp_preds) + 1e-8)
    return np.argmax(np.random.multinomial(1, probs, 1))

def generate_text(seed, length, temp, model, vocab):
    generated = seed
    seq_length = vocab['seq_length']
    
    for _ in range(length):
        # 截取或填充序列
        tokens = list(generated)[-seq_length:] if vocab['is_poetry'] \
                else jieba.lcut(generated)[-seq_length:]
        
        # 填充序列到固定长度
        if len(tokens) < seq_length:
            tokens = ['<PAD>'] * (seq_length - len(tokens)) + tokens
        
        # 创建输入向量
        x = np.zeros((1, seq_length, vocab['vocab_size']), dtype=np.bool_)
        for t, word in enumerate(tokens[-seq_length:]):  # 取最后seq_length个token
            if word in vocab['word2idx']:
                x[0, t, vocab['word2idx'][word]] = 1
            else:  # 处理未登录词
                x[0, t, vocab['word2idx'].get('<UNK>', 0)] = 1
        
        # 预测时添加异常处理
        try:
            preds = model.predict(x, verbose=0)[0]
            next_idx = sample_with_temp(preds, temp)
            generated += vocab['idx2word'][str(next_idx)]
        except Exception as e:
            st.error(f"生成出错: {str(e)}")
            break
    
    return generated

if __name__ == "__main__":
    main()