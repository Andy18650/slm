import re


WORD_PATTERN = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)
UNK_TOKEN = "<unk>"


def maybe_lower(text: str, lowercase: bool) -> str:
    return text.lower() if lowercase else text


def train_char_tokenizer(texts: dict[str, str], lowercase: bool) -> dict:
    combined = "".join(maybe_lower(text, lowercase) for text in texts.values())
    tokens = sorted(set(combined))
    stoi = {token: index for index, token in enumerate(tokens)}
    itos = {index: token for token, index in stoi.items()}
    return {"type": "char", "lowercase": lowercase, "stoi": stoi, "itos": itos}


def train_word_tokenizer(texts: dict[str, str], lowercase: bool, min_frequency: int) -> dict:
    counts: dict[str, int] = {}
    for text in texts.values():
        for token in WORD_PATTERN.findall(maybe_lower(text, lowercase)):
            counts[token] = counts.get(token, 0) + 1

    tokens = [UNK_TOKEN]
    tokens.extend(sorted(token for token, count in counts.items() if count >= min_frequency))
    stoi = {token: index for index, token in enumerate(tokens)}
    itos = {index: token for token, index in stoi.items()}
    return {
        "type": "word",
        "lowercase": lowercase,
        "min_frequency": min_frequency,
        "unk_token": UNK_TOKEN,
        "stoi": stoi,
        "itos": itos,
    }


def train_bpe_tokenizer(texts: dict[str, str], lowercase: bool, vocab_size: int) -> dict:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=[UNK_TOKEN])
    tokenizer.train_from_iterator((maybe_lower(text, lowercase) for text in texts.values()), trainer)
    return {
        "type": "bpe",
        "lowercase": lowercase,
        "vocab_size_target": vocab_size,
        "unk_token": UNK_TOKEN,
        "tokenizer_json": tokenizer.to_str(),
    }


def train_tokenizer(
    texts: dict[str, str],
    tokenizer_type: str,
    lowercase: bool,
    vocab_size: int,
    min_frequency: int,
) -> dict:
    if tokenizer_type == "char":
        return train_char_tokenizer(texts, lowercase)
    if tokenizer_type == "word":
        return train_word_tokenizer(texts, lowercase, min_frequency)
    if tokenizer_type == "bpe":
        return train_bpe_tokenizer(texts, lowercase, vocab_size)
    raise ValueError(f"Unsupported tokenizer: {tokenizer_type}")


def encode_text(text: str, tokenizer_meta: dict) -> list[int]:
    text = maybe_lower(text, tokenizer_meta.get("lowercase", False))
    tokenizer_type = tokenizer_meta["type"]

    if tokenizer_type == "char":
        stoi = tokenizer_meta["stoi"]
        missing = sorted(set(text) - set(stoi))
        if missing:
            raise ValueError(f"Text contains characters not in the training vocabulary: {missing}")
        return [stoi[char] for char in text]

    if tokenizer_type == "word":
        stoi = tokenizer_meta["stoi"]
        unk_id = stoi[tokenizer_meta["unk_token"]]
        return [stoi.get(token, unk_id) for token in WORD_PATTERN.findall(text)]

    if tokenizer_type == "bpe":
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_str(tokenizer_meta["tokenizer_json"])
        return tokenizer.encode(text).ids

    raise ValueError(f"Unsupported tokenizer: {tokenizer_type}")


def decode_tokens(token_ids: list[int], tokenizer_meta: dict) -> str:
    tokenizer_type = tokenizer_meta["type"]

    if tokenizer_type in {"char", "word"}:
        itos = {int(key): value for key, value in tokenizer_meta["itos"].items()}
        return "".join(itos[int(token_id)] for token_id in token_ids)

    if tokenizer_type == "bpe":
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_str(tokenizer_meta["tokenizer_json"])
        return tokenizer.decode([int(token_id) for token_id in token_ids])

    raise ValueError(f"Unsupported tokenizer: {tokenizer_type}")


def tokenizer_vocab_size(tokenizer_meta: dict) -> int:
    tokenizer_type = tokenizer_meta["type"]
    if tokenizer_type in {"char", "word"}:
        return len(tokenizer_meta["stoi"])
    if tokenizer_type == "bpe":
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_str(tokenizer_meta["tokenizer_json"])
        return tokenizer.get_vocab_size()
    raise ValueError(f"Unsupported tokenizer: {tokenizer_type}")
