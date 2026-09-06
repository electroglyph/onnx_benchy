"""Batching tests use a stub tokenizer (no network)."""


class StubTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, truncation=False,
                 verbose=True):
        # word-length encoding: deterministic, non-empty
        return {"input_ids": [len(w) + 1 for w in text.split()] or [7]}


def test_pack_dense_no_padding():
    from onnx_benchy.batching import BatchStream, pre_tokenize

    tok = StubTokenizer()
    texts = ["hello world foo bar", "a b c d e f g", "one two three four five"]
    tokenized = pre_tokenize(tok, texts, pack=True)
    stream = BatchStream(tokenized, batch_size=2, context_size=4, pack=True,
                         pad_id=0, texts=texts, tokenizer=tok,
                         shuffle=False, seed=0)
    ids, mask, nonpad = stream.next_batch()
    assert ids.shape == (2, 4)
    assert nonpad == 8  # dense: B*C
    assert mask.sum() == 8


def test_no_pack_counts_nonpad_only():
    from onnx_benchy.batching import BatchStream, pre_tokenize

    tok = StubTokenizer()
    texts = ["hi", "a b c d e f g h i j k l"]
    tokenized = pre_tokenize(tok, texts, pack=False)
    stream = BatchStream(tokenized, batch_size=2, context_size=8, pack=False,
                         pad_id=0, texts=texts, tokenizer=tok,
                         shuffle=False, seed=0)
    ids, mask, nonpad = stream.next_batch()
    assert ids.shape == (2, 8)
    assert nonpad == int(mask.sum()) < 16


def test_wrap_around_epochs():
    from onnx_benchy.batching import BatchStream, pre_tokenize

    tok = StubTokenizer()
    texts = ["a b c d"]
    tokenized = pre_tokenize(tok, texts, pack=True)
    stream = BatchStream(tokenized, batch_size=1, context_size=2, pack=True,
                         pad_id=0, texts=texts, tokenizer=tok,
                         shuffle=False, seed=0)
    for _ in range(5):  # only 2 blocks available; must wrap, not crash
        stream.next_batch()
    assert stream.epoch >= 1
