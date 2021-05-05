# pyvgram
🍺 Python implementation on vgram tokenization

VGram is a tokenizer construction algorithm that optimizes the code length of the text.
It can be used to tokenize text like BPE (Sennrich et al., 2016).

Read more in our CIKM'18 paper [Construction of Efficient V-Gram Dictionary for Sequential Data Analysis](https://dl.acm.org/doi/10.1145/3269206.3271789).

## Install
```bash
pip install pyvgram
```

## Examples

### 1. Quickstart

Let's train tokenizer with size `10000` on `file.txt` content and encodes some string.

```python
from vgram import VGramTokenizer

tokenizer = VGramTokenizer(10000)
tokenizer.train(["file.txt"])
ids = tokenizer.encode("hello world")
```

### 2. Save and load

```python
from vgram import VGramTokenizer

tokenizer = VGramTokenizer(10000)
tokenizer.train(["file.txt"])
ids1 = tokenizer.encode("hello world")

tokenizer.save_pretrained("vgram.tokenizer")
loaded_tokenizer = VGramTokenizer.from_pretrained("vgram.tokenizer")
ids2 = loaded_tokenizer.encode("hello world")

assert tokenizer == loaded_tokenizer
assert ids1 == ids2
```

### 3. Learn from raw text

You can learn a tokenizer from raw text by `fit` method.

```python
from vgram import VGramTokenizer

tokenizer = VGramTokenizer(10000)
tokenizer.fit(" ".join(["hello world"] * 1000))
ids = tokenizer.encode("hello world")
```

Also, you can specify iters num if you want to learn more by bootstrap sampling.

```python
from vgram import VGramTokenizer

tokenizer = VGramTokenizer(10000)
tokenizer.fit("hello world", iters=1000))
ids = tokenizer.encode("hello world")
```

### 4. Learn multiple times

You can learn a tokenizer on one dataset and then finetune on another 
by multiple calls of `fit` or `train` methods.

```python
from vgram import VGramTokenizer, SplitLevel

tokenizer = VGramTokenizer(200, split_level=SplitLevel.NONE)
tokenizer.fit(["hello", "hello world"], iters=10000))
assert len(tokenizer.encode("hello world")) == 1
assert len(tokenizer.encode("pip install pyvgram")) > 1

tokenizer.fit("pip install pyvgram", iters=10000))
assert len(tokenizer.encode("hello world")) > 1
assert len(tokenizer.encode("pip install pyvgram")) == 1
```

After finetuning `tokenizer.encode("hello world")` codes by symbols 
into `['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']`  
because in finetune dataset it's not meaningful sequence.

### 5. Vocabulary

```python
from vgram import VGramTokenizer, SplitLevel

tokenizer = VGramTokenizer(10000, split_level=SplitLevel.LINE)
tokenizer.fit(" ".join(["hello world"] * 1000))
print("Vocabulary:", tokenizer.get_vocab())
# Vocabulary: ['h', 'hello world', 'e', 'l', 'o', ' ', 'w', 'r', 'd', '\n']
print("Vocab size:", tokenizer.vocab_size())
# Vocab size: 10
```

### 6. Learn with another split-level

The most of bpe-like tokenization libraries split one word to the pieces.
`pyvgram` support different levels of splitting, 
so you can split whole line in to pieces which consist of few words if they are frequent enough.
It's useful for analyzing vocabulary to find patterns in data.

Default split-level is `WORD`, but you can also use `LINE` and `NONE`.
```python
from vgram import VGramTokenizer, SplitLevel

text = "\n".join(["hello world"] * 10000)

tokenizer = VGramTokenizer(200, split_level=SplitLevel.WORD)
tokenizer.fit(text)
print(tokenizer.get_vocab())
# ['h', 'hello', 'e', 'l', 'o', ' ', ' world', 'w', 'r', 'd', '\n']

tokenizer = VGramTokenizer(200, split_level=SplitLevel.LINE)
tokenizer.fit(text)
print(tokenizer.get_vocab())
# ['h', 'hello world', 'e', 'l', 'o', ' ', 'w', 'r', 'd', '\n']
```

`SplitLevel.NONE` not split text and handle it like one sequence. 
Its bad idea to pass very few texts in such case, 
but if you have many pre-splited texts, it's a good choice 
```python
from vgram import VGramTokenizer, SplitLevel

texts = ["hello world"] * 10000

tokenizer = VGramTokenizer(200, split_level=SplitLevel.NONE)
tokenizer.fit(texts)
print(tokenizer.get_vocab())
# ['h', 'hello world', 'e', 'l', 'o', ' ', 'w', 'r', 'd']
```