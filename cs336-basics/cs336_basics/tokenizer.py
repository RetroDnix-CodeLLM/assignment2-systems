import os
import regex as re
from typing import Iterable, Iterator
from time import time
from tqdm import tqdm
from pickle import dump, load
from multiprocessing import Pool, cpu_count
from collections import Counter

from cs336_basics.utils import UnionFindSet, increaseD, decreaseD, appendD, removeD

class BPETrainer():
    pretokenizePAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    def __init__(self, corpus:str, special_tokens: list[str] = ["<|endoftext|>"]):
        self.vocab = {}
        for i in range(256):
            self.vocab[bytes([i,])] = i
        
        for sp_token in special_tokens:
            self.vocab[sp_token.encode("utf-8")] = len(self.vocab)
        
        self.corpus = []
        self.frequency = {}
        self.merges = []

        PAT = '|'.join(map(re.escape, special_tokens))
        with open(corpus, "r", encoding="utf-8") as f:
            self.corpus = re.split(PAT, f.read())

    @staticmethod
    def pre_tokenize(text: str):
        """
        对分块之后的语料库进行预分词

        预分词之后，不考虑合并垮两个“pre-tokenized-token”的字节

        Args:
            text (str): 需要预分词的文本
        
        Returns:
            frequency (dict[str:int]): 预分词后的token列表
        """
        return re.finditer(BPETrainer.pretokenizePAT, text, re.UNICODE)
    
    def pre_tokenize_corpus(self):
        """
        对语料库进行预分词
        """
        for text in tqdm(self.corpus, desc="Pre-tokenizing corpus"):
            for result in BPETrainer.pre_tokenize(text):
                token = result.group(0)
                self.frequency[token] = self.frequency.get(token, 0) + 1
        
        print(f"Pre-tokenization complete. Found {len(self.frequency)} unique tokens.")

    @staticmethod
    def _process_text_chunk(text_chunk: str):
        """
        处理单个文本块的辅助函数，用于并行处理
        
        Args:
            text (str): 需要处理的文本块
        
        Returns:
            dict: 该文本块中token的频率字典
        """
        id = text_chunk[0]
        chunk = text_chunk[1]
        frequency = Counter()
        i = 0
        for text in chunk:
            for result in re.finditer(BPETrainer.pretokenizePAT, text, re.UNICODE):
                token = result.group(0)
                frequency[token] += 1
            if i % 10000 == 0:
                print(f"Process {id}: {i} / {len(chunk)}")
            i += 1
        return frequency

    def parallel_pre_tokenize_corpus(self, num_processes=None):
        """
        对语料库进行并行预分词
        
        Args:
            num_processes (int, optional): 使用的进程数。如果为None，则使用CPU核心数
        """
        if num_processes is None:
            num_processes = cpu_count()
        
        print(f"Starting parallel pre-tokenization using {num_processes} processes...")
        
        # 初始化频率字典
        self.frequency = Counter()
        self.chunked_corpus = [self.corpus[i::num_processes] for i in range(num_processes)]
        
        # 使用多进程池处理语料库
        with Pool(num_processes) as pool:
            for result in pool.imap_unordered(BPETrainer._process_text_chunk, enumerate(self.chunked_corpus)):
                self.frequency += result
        
        print(f"Parallel pre-tokenization complete. Found {len(self.frequency)} unique tokens.")

    def train_bpe(self, maximum_vocab_size: int = 10000):
        """
        在语料库上合并BPE字节对。

        Args:
            maximum_vocab_size (int): 训练BPE的最大词汇表大小，默认为10000

        """
        appearances = {}    # apprearance[pair]: 表示字符对pair出现的频率
        position = {}       # position[pair]: 表示字符对pair(a,b)出现的位置（使用b的首字节作为标记）
        value = {}          # value[(i,j,k)]: 表示位置(i,j,k)的字节对的值（首字节标记）
        ufs = {}
        
        maxn = 0
        maxp = None
        frequency = self.frequency

        if len(self.vocab) < maximum_vocab_size:
            pbar = tqdm(total=len(frequency), desc="BPE Training(Pre-processing)...")
            for token in frequency:
                bs = [(b, ) for b in token.encode("utf-8")]
                l = len(bs)
                s = ufs[token] = UnionFindSet(size = l)
                v = value[token] = {}
                for k in range(l):
                    v[k] = bs[k]

                    if k < l - 1:
                        pair = (bs[k], bs[k + 1])
                        increaseD(appearances, pair, frequency[token])
                        appendD(position, pair, (token, k))
                        if appearances[pair] > maxn or (appearances[pair] == maxn and pair > maxp):
                            maxn = appearances[pair]
                            maxp = pair
                pbar.update(1)
            pbar.close()
        
        pbar = tqdm(total=maximum_vocab_size - len(self.vocab), desc="BPE Training(Merging byte pairs)...")
        while len(self.vocab) < maximum_vocab_size:
            current_vocab_size = len(self.vocab)
            if maxp is not None:
                bMaxPair = maxp[0] + maxp[1]
                self.merges.append((bytes(maxp[0]), bytes(maxp[1])))
                self.vocab[bytes(bMaxPair)] = len(self.vocab)

                pbar.update(len(self.vocab) - current_vocab_size) 
                
                merged = {}
                pos = list(position[maxp])
                for token, k1 in pos:
                    if (token, k1) in merged:
                        continue
                    s = ufs[token]
                    v = value[token]
                    k2 = k1 + s.getSize(k1)
                    sizek2 = s.getSize(k2)
                    s.union(k1, k2)
                    v[k1] = bMaxPair
                    
                    if s.has(k1 - 1):
                        k0 = s.find(k1 - 1)
                        bPre = v[k0]
                        increaseD(appearances, (bPre, bMaxPair), frequency[token])
                        appendD(position, (bPre, bMaxPair), (token, k0))
                        
                        decreaseD(appearances, (bPre, maxp[0]), frequency[token])
                        removeD(position, (bPre, maxp[0]), (token, k0))
                        if (bPre, maxp[0]) == maxp:
                            merged[(token, k0)] = True

                    if s.has(k2 + sizek2):
                        k3 = k2 + sizek2
                        bSuf = v[k3]
                        increaseD(appearances, (bMaxPair, bSuf), frequency[token])
                        appendD(position, (bMaxPair, bSuf), (token, k1))

                        decreaseD(appearances, (maxp[1], bSuf), frequency[token])
                        removeD(position, (maxp[1], bSuf), (token, k2))
                        if (maxp[1], bSuf) == maxp:
                            merged[(token, k2)] = True
                        
                appearances.pop(maxp)
                position.pop(maxp)
                v.pop(k2)

                maxn = 0
                maxp = None
                for pair in appearances:
                    if appearances[pair] > maxn or (appearances[pair] == maxn and pair > maxp):
                        maxn = appearances[pair]
                        maxp = pair

        pbar.close()
        print("BPE training complete.")

    def save_vocab(self, output_path: str | os.PathLike):
        """
        保存BPE词汇表到指定路径

        Args:
            output_path (str | os.PathLike): 输出路径
        """
        with open(output_path, "wb") as f:
            dump(self.vocab, f)
        print(f"Vocabulary saved to {output_path}")
    
    def save_merges(self, output_path: str | os.PathLike):
        """
        保存BPE合并规则到指定路径

        Args:
            output_path (str | os.PathLike): 输出路径
        """
        with open(output_path, "wb") as f:
            dump(self.merges, f)
        print(f"Merges saved to {output_path}")
    
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
):
    tokenizer = BPETrainer(corpus = input_path, special_tokens=special_tokens)
    s = time()
    tokenizer.parallel_pre_tokenize_corpus(4)
    # tokenizer.pre_tokenize_corpus()
    d = time()
    print(f"Pre-tokenization took {d - s:.2f} seconds.")
    tokenizer.train_bpe(maximum_vocab_size=vocab_size)
    print(f"BPE training took {time() - d:.2f} seconds.")
    reversed_vocab = {v: k for k, v in tokenizer.vocab.items()}
    return reversed_vocab, tokenizer.merges

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept the following parameters:  
        Args:
            vocab: dict[int, bytes] | str 
            merges: list[tuple[bytes, bytes]] | str
            special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges
        
        if isinstance(vocab, str):
            with open(vocab, "rb") as f:
                self.vocab = load(f)
        if isinstance(merges, str):
            with open(merges, "rb") as f:
                self.merges = load(f)

        self.id2bytes = {v: k for k, v in self.vocab.items()}
        self.word2token = {}
        self.special_tokens = special_tokens if special_tokens is not None else [r"[\r\n]$",]
        for i, sp_token in enumerate(self.special_tokens):
            for j in range(i + 1, len(self.special_tokens)):
                if sp_token in self.special_tokens[j]:
                    self.special_tokens[i] = self.special_tokens[j]
                    self.special_tokens[j] = sp_token
    
    @staticmethod
    def from_files(vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class  method that constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens. 
        Args:
            vocab_filepath: str  
            merges_filepath: str  
            special_tokens: list[str] | None = None  
        """
        with open(vocab_filepath, "rb") as f:
            vocab = load(f)
        
        with open(merges_filepath, "rb") as f:
            merges = load(f)
        
        return BPETokenizer(vocab, merges, special_tokens)

    def _encode_sent(self, sentence: str) -> list[int]:
        result = []
        words = BPETrainer.pre_tokenize(sentence)
        for word in words:
            word = word.group(0)
            if word in self.word2token:
                result.extend(self.word2token[word])
            else:
                bword = [bytes((i,)) for i in word.encode("utf-8")]
                for pair in self.merges:
                    i = 0
                    maxi = len(bword) - 1
                    while i < maxi:
                        if bword[i] == pair[0] and bword[i + 1] == pair[1]:
                            bword[i:i + 2] = [pair[0] + pair[1]]
                            maxi = len(bword) - 1
                        i += 1
                    if len(bword) <= 1:
                        break
                ids = [self.vocab[b] for b in bword if b in self.vocab]
                self.word2token[word] = ids
                result.extend(ids)
        return result
    
    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        PAT = '|'.join(map(re.escape, self.special_tokens))
        result = []
        matches = re.finditer(PAT, text)
        total = len(text)
        lastpos = pos = 0
        for match in matches:
            result.extend(self._encode_sent(text[pos:match.start()]))
            result.append(self.vocab[match.group(0).encode("utf-8")])
            pos = match.end()
            if pos - lastpos > 1000000:
                print(f"Processed {pos}/{total} characters.")
                lastpos = pos
        result.extend(self._encode_sent(text[pos:]))
        return result
    
    def encode_iterable(self, iterable: Iterable[str]) ->  Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-efficient tokenization of large files that we cannot directly load into memory.  
        """
        buffer = ""
        for text in iterable:
            buffer += text
            PAT = '|'.join(map(re.escape, self.special_tokens))
            matches = re.finditer(PAT, buffer)
            for match in matches:
                yield from self._encode_sent(buffer[:match.start()])
                yield self.vocab[match.group(0).encode("utf-8")]
                buffer = buffer[match.end():]
        yield from self._encode_sent(buffer)
    
    def parallel_tokenize_txt(
        self,
        file_path: str,
        num_workers: int,
    ):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Loaded {len(lines)} lines from {file_path}")

        chunk_size = len(lines) // num_workers
        chunks = ["\n".join(lines[i:i + chunk_size]) for i in range(0, len(lines), chunk_size)]
        
        print(f"Tokenizing with {num_workers} workers...")

        tokenized_data = []
        with Pool(num_workers) as pool:
            results = list(pool.imap(self.encode, chunks))

        for result in results:
            tokenized_data.extend(result)

        return tokenized_data

    
    def decode(self, ids: list[int]|Iterator[int]|int) -> str:
        """
        Decode a sequence of token IDs into text.  To test your Tokenizer against our provided tests, you will first need to implement the test adapter at [adapters.get_tokenizer]. Then, run uv run pytest tests/test_tokenizer.py. Your implementation should be able to pass all tests.
        """
        if isinstance(ids, int):
            ids = [ids]
        byte_s = [self.id2bytes[i] for i in ids if i in self.id2bytes]
        text = b"".join(byte_s).decode("utf-8", errors="replace")
        return text

if __name__ == "__main__":
    trainer = BPETrainer("data/baby_data.txt", special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    trainer.parallel_pre_tokenize_corpus(4)
    trainer.train_bpe(maximum_vocab_size=300)

    tokenizer = BPETokenizer(trainer.vocab, trainer.merges)
    print(trainer.vocab)
    with open("data/baby_data.txt") as f:
        ids = tokenizer.encode_iterable(f)
        print(tokenizer.decode(ids))

    # ids = tokenizer.encode("")
    # print(ids)
    # print(tokenizer.decode(ids))