import os


def get_text(text_path):
    text = ''
    files = os.listdir(text_path)
    for file in files:
        with open(text_path + '/' + file, 'r', encoding='utf-8') as f:
            text = text + f.read() + '\n'
    return text

def get_temp_merges(text_code):
    temp_merges = {}
    for i in range(len(text_code) - 1):
        if (text_code[i], text_code[i + 1]) not in temp_merges:
            temp_merges[(text_code[i], text_code[i + 1])] = 1
        else:
            temp_merges[(text_code[i], text_code[i + 1])] += 1
    return temp_merges

def get_new_text_code(text_code, pair, idx):
    new_text_code = []
    i = 0
    while i < len(text_code):
        if (i < len(text_code) - 1) and (text_code[i], text_code[i + 1]) == pair[0]:
            new_text_code.append(idx)
            i += 2
        else:
            new_text_code.append(text_code[i])
            i += 1
    return new_text_code


def encode(text, merges):
    text_code = list(text.encode('utf-8'))
    while len(text_code)>2:
        temp_merges = get_temp_merges(text_code)
        min_pair = max(temp_merges.items(), key=lambda item: item[1])
        if min_pair[0] not in merges:
            return text_code
        else:
            text_code = get_new_text_code(text_code, pair=min_pair, idx=merges[min_pair[0]])

def decode(tokens, vocab):
    tokens = b"".join(vocab[idx] for idx in tokens)
    text = tokens.decode("utf-8", errors="replace")
    return text

def main():
    text_path = r"D:\download\BaiduNetdiskDownload\第十四周 大语言模型RAG\week14 大语言模型相关第四讲\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\Heroes"
    text = get_text(text_path)
    text_code = list(text.encode('utf-8'))
    merges = {}
    vocab_size = 500
    for i in range(vocab_size-256):
        temp_merges = get_temp_merges(text_code)
        max_pair = max(temp_merges.items(), key=lambda item: item[1])
        merges[max_pair[0]] = 256+i
        text_code = get_new_text_code(text_code, pair=(max_pair[0], max_pair[1]), idx=256+i)
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for pair, idx in merges.items():
        vocab[idx] = vocab[pair[0]]+vocab[pair[1]]

    string = "矮人直升机"
    encode_tokens = encode(string, merges)
    print(encode_tokens)
    decode_tokens = decode(encode_tokens, vocab)
    print(decode_tokens)
#[424, 174, 228, 186, 186, 231, 155, 180, 229, 141, 135, 230, 156, 186]
#矮人直升机
main()
