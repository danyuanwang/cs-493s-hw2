class TokenizerOutput:
    def __init__(self, input_ids):
        self.input_ids = input_ids

class CharTokenizer:
    def __init__(self):
        self.vocab = ['<PAD>', '<BOS>', '<EOS>'] + list('0123456789+-=/ ')
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.pad_token = '<PAD>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.pad_token_id = self.stoi[self.pad_token]

    def encode(self, text, add_bos_eos=True):
        tokens = []
        if add_bos_eos:
            tokens.append(self.stoi[self.bos_token])
        tokens.extend(self.stoi[ch] for ch in text)
        if add_bos_eos:
            tokens.append(self.stoi[self.eos_token])
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        chars = []
        for id in token_ids:
            ch = self.itos[id]
            if skip_special_tokens and ch in {'<PAD>', '<BOS>', '<EOS>'}:
                continue
            chars.append(ch)
        return ''.join(chars)
    
    def __call__(self, text, add_bos_eos=True):
        input_ids = self.encode(text, add_bos_eos=add_bos_eos)
        return TokenizerOutput(input_ids=input_ids)