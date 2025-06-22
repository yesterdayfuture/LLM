
import re
from collections import defaultdict, Counter
import torch
from typing import List, Dict, Tuple

class BBPETokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        # 基础词汇表: 256字节 + 特殊token ，计算机中所有数据（包括文本）最终存储为 ​​0-255的整数序列​​（1字节=8比特）
        self.token_to_id = {bytes([i]): i for i in range(256)}
        self.token_to_id[b'</w>'] = 256
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.merges = []  # 存储合并规则
        self.special_tokens = {'<pad>': 257, '<unk>': 258, '<bos>': 259, '<eos>': 260}
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """添加特殊token到词汇表"""
        for token, idx in self.special_tokens.items():
            self.token_to_id[token.encode('utf-8')] = idx
            self.id_to_token[idx] = token.encode('utf-8')
    
    def _preprocess(self, corpus: List[str]) -> Dict[str, int]:
        """文本预处理：分割单词并统计频率"""
        word_freqs = Counter()
        for text in corpus:
            # 保留空格作为独立单词
            words = re.split(r'(\s)', text.strip())
            word_freqs.update(w for w in words if w)
        return word_freqs
    
    def _initialize_splits(self, word_freqs: Dict[str, int]) -> Dict[str, List[bytes]]:
        """初始化单词的分割状态"""
        splits = {}
        for word, freq in word_freqs.items():
            # 转换为字节并添加结束符
            byte_sequence = list(word.encode('utf-8')) # 将当前 单词（单个中文字或中文字符串） 转换为字节，然后变为字节对应的索引
            splits[word] = [bytes([b]) for b in byte_sequence] + [b'</w>']
        return splits
    
    def train(self, corpus: List[str]):
        """BBPE训练过程"""
        # 1. 预处理语料
        word_freqs = self._preprocess(corpus) # 统计每个单词的频率
        splits = self._initialize_splits(word_freqs) # 获取每个单词 对应的字节列表
        
        # 2. 迭代合并字节对
        while len(self.token_to_id) < self.vocab_size:
            # 统计字节对频率
            pair_freqs = defaultdict(int)
            for word, tokens in splits.items():
                for i in range(len(tokens) - 1):
                    # 跳过含</w>的字节对
                    if b'</w>' in (tokens[i], tokens[i+1]):
                        continue
                    pair = (tokens[i], tokens[i+1])
                    pair_freqs[pair] += word_freqs[word]
            
            if not pair_freqs:
                break
                
            # 3. 合并最高频字节对
            best_pair = max(pair_freqs, key=pair_freqs.get)
            new_token = best_pair[0] + best_pair[1]
            
            # 4. 更新词汇表
            new_id = len(self.token_to_id)
            self.token_to_id[new_token] = new_id
            self.id_to_token[new_id] = new_token
            self.merges.append(best_pair)
            
            # 5. 更新所有单词的分割状态
            for word in splits:              # 遍历语料库中每个单词的分割状态
                new_tokens = []              # 初始化新token列表，存储合并后的分割结果
                i = 0                        # 初始化索引，用于遍历当前单词的token序列
                
                # 遍历当前单词的每个token
                while i < len(splits[word]):
                    # 检查条件：
                    # 1. 当前token不是最后一个（避免索引越界）
                    # 2. 当前token与下一个token组成的字节对等于本轮最优对 best_pair
                    if i < len(splits[word]) - 1 and \
                    (splits[word][i], splits[word][i+1]) == best_pair:
                        
                        # 将合并后的新token加入结果列表
                        new_tokens.append(new_token)  # new_token = best_pair[0] + best_pair[1]
                        i += 2               # 跳过已合并的两个token，直接处理后续位置
                    
                    # 若不满足合并条件：
                    else:
                        # 保留当前token（无需合并）
                        new_tokens.append(splits[word][i])
                        i += 1               # 移至下一个token
                
                # 更新当前单词的分割状态为合并后的结果
                splits[word] = new_tokens
    
    def encode(self, text: str) -> List[int]:
        """文本编码为token ID序列"""
        # 分割单词（保留空格）
        words = re.split(r'(\s)', text.strip())
        ids = []
        
        for word in words:
            if not word:
                continue
            # 初始字节分割
            tokens = [bytes([b]) for b in word.encode('utf-8')] + [b'</w>']
            
            # 应用所有合并规则
            for merge_pair in self.merges:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and \
                       (tokens[i], tokens[i+1]) == merge_pair:
                        new_token = merge_pair[0] + merge_pair[1]
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            
            # 转换为ID（处理未知token）
            for token in tokens:
                if token in self.token_to_id:
                    ids.append(self.token_to_id[token])
                else:
                    # 回溯拆分未知token
                    for b in token:
                        byte_token = bytes([b])
                        ids.append(self.token_to_id.get(byte_token, self.special_tokens['<unk>']))
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """token ID序列解码为文本"""
        byte_sequence = bytearray()
        words = []
        
        for id_val in ids:
            token = self.id_to_token.get(id_val)
            
            if token == b'</w>':  # 单词结束
                words.append(byte_sequence.decode('utf-8', errors='replace'))
                byte_sequence.clear()
            elif token in self.special_tokens.values():  # 特殊token
                continue
            else:
                byte_sequence.extend(token)
        
        # 处理最后一个单词
        if byte_sequence:
            words.append(byte_sequence.decode('utf-8', errors='replace'))
        
        return ''.join(words)


def saveMergeVocab(merge_vocab_list):
        with open('./merge_vocab.txt', 'w') as f:
            for item in merge_vocab_list:
                f.write(f"{item}\n")


# 示例用法
if __name__ == "__main__":
    # 1. 准备训练语料
    corpus = [
        "Byte-level BPE handles multilingual text.",
        "例如中文和English混合的场景",
        "Привет! こんにちは! 🚀"
    ]
    
    # 2. 训练BBPE分词器
    tokenizer = BBPETokenizer(vocab_size=500)
    tokenizer.train(corpus)

    # 3. 保存合并后的词汇表
    saveMergeVocab(tokenizer.token_to_id)
    
    # 3. 编码测试
    text = "BBPE处理emoji: 😊👍🚀"
    encoded_ids = tokenizer.encode(text)
    print(f"编码结果: {encoded_ids}")
    
    # 4. 解码验证
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"解码结果: {decoded_text}")





