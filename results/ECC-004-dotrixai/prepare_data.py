"""ECC-004 data: OpenWebText -> byte-level BPE with 8,192 tokens trained on training text only -> uint16 token files with SHA-256.

Runs as a Kaggle CPU kernel (internet on), or anywhere with about 10 GB of free disk and the packages numpy, pyarrow and tokenizers.

    python prepare_data.py --out OUT_DIR [--work TMP_DIR]                # as a Kaggle kernel: no arguments, writes to /kaggle/working
    python prepare_data.py --out OUT_DIR --smoke --shard-dir DIR      # small local test on a few parquet files, no download, no hash check

Source: Hugging Face dataset Skylion007/openwebtext (CC0-1.0), revision 79d93d786212f7344586290adb811d4ae6a1762c, the parquet files
plain_text/train-000NN-of-00080.parquet. Every downloaded file is checked against the SHA-256 listed below (the dataset's LFS ids).

Split (by file, decided before any text was read):
  - training text: files 00000 to 00011, all documents, in file and row order;
  - validation text: file 00079, the first 12,000 documents that are not exact copies of a training document.
  - The tokenizer is trained on the first 300,000 training documents only. Validation text is never used for the tokenizer.
Encoding: every document is encoded on its own and followed by the separator token <|endoftext|> (id 0). Byte-level BPE, no
normalization, no prefix space, so decoding gives the exact UTF-8 text back (checked on a sample).

Outputs (in OUT_DIR):
  train.bin, val.bin     uint16 token ids
  token_bytes.npy        UTF-8 bytes per token id (the separator counts as 1 byte, standing in for a document break); used for bits per byte
  tokenizer.json         the trained tokenizer
  meta.json              hashes, counts, source revision, file hashes and package versions (read by ecc004.py)
"""
import argparse
import hashlib
import json
import os
import platform
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

REPO, REV = 'Skylion007/openwebtext', '79d93d786212f7344586290adb811d4ae6a1762c'
TRAIN_FILES, VAL_FILE = list(range(0, 12)), 79
VAL_DOCS, TOK_DOCS, VOCAB, EOS = 12_000, 300_000, 8192, '<|endoftext|>'
SHA = {
    0: 'caed9f4b7053d7cd4d1a13ce9ec9224d84a3bba1f11579193562a7e31ebe656e',
    1: '70307b7749e10cb0bfb895fde024dd82d96e6bb2710bca1886b952abe45bf3ea',
    2: '0d02a67f73464d383c8df2059f27b36b4b1652c855051be3c65930baef12bd2d',
    3: '437833e29326c3d88792e834d36368d2c4836e5c5f3535233a88f3cf7dd46aad',
    4: '7b3cef26e2d4e2b041e6e13a0102cb4d03277e8c6418b8ff1b865544a4695de3',
    5: '2d3200a936b021c9abafdda6714e4d678387e8854e56abaac5d2b62882fc6044',
    6: '76c0b96ce0250c3a32f2e8878172a4b79ac7970fede78edfcfc4d98d87ec5a6c',
    7: 'd6088d5452e4cd840a6a2d52497ba6b92d003a62394788789e246fd58090863d',
    8: '9d738ba5d171bb2a0b28179f9a5335411c9846140086c4a1f170fb1fd0d86d12',
    9: '51e193a86a363394bdf5344918980d1e873150bf6ded659673375a4f75397f1e',
    10: 'c5e950cdeb9a958580aa279dacd44a9a45507c8777a9105c89d0d605f8f01e11',
    11: '699a4f4fc791d4ba9e6487ca91b63faa59a831cd5975aeaf8260451af9bb432f',
    79: '80f15bb3dc64287827248d2a7a999ec986ecf15f07a79ed8c2e2b366e8046ae2',
}


def name(i):
    return f'plain_text/train-{i:05d}-of-00080.parquet'


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def download(i, work, log):
    dst = work / Path(name(i)).name
    if not dst.exists():
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{name(i)}'
        for attempt in range(5):
            try:
                tmp = dst.with_suffix('.part')
                with urllib.request.urlopen(url, timeout=120) as r, open(tmp, 'wb') as f:
                    while True:
                        b = r.read(1 << 22)
                        if not b:
                            break
                        f.write(b)
                os.replace(tmp, dst); break
            except Exception as ex:                                               # noqa: BLE001
                log(f'download {name(i)} attempt {attempt + 1} failed: {ex}'); time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f'could not download {name(i)}')
    h = sha256_file(dst)
    assert h == SHA[i], f'{name(i)}: sha256 {h} != {SHA[i]}'
    return dst, h


def docs(path, batch=4096):
    for b in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=['text']):
        yield b.column(0).to_pylist()


def byte_decoder():
    """Inverse of the GPT-2 byte-to-unicode map used by the ByteLevel pre-tokenizer."""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main():
    ap = argparse.ArgumentParser()
    on_kaggle = bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE'))                         # as a Kaggle kernel the script runs without arguments
    ap.add_argument('--out', default='/kaggle/working' if on_kaggle else None, required=not on_kaggle); ap.add_argument('--work', default='/tmp/ecc004_owt')
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--shard-dir')
    a = ap.parse_args()
    t0 = time.time(); out, work = Path(a.out), Path(a.work); out.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(f'[{time.time() - t0:7.0f} s] {s}', flush=True)
    val_docs, tok_docs, train_files = (VAL_DOCS, TOK_DOCS, TRAIN_FILES) if not a.smoke else (150, 400, [0, 1])
    meta = dict(source=dict(repo=REPO, revision=REV, license='CC0-1.0', train_files=[name(i) for i in train_files], val_file=name(VAL_FILE),
                            file_sha256={}), rules=dict(val_docs=val_docs, tokenizer_docs=tok_docs, vocab=VOCAB, separator=EOS,
                                                         separator_bytes=1, smoke=a.smoke),
                versions=dict(python=platform.python_version(), numpy=np.__version__, pyarrow=pyarrow.__version__,
                              tokenizers=tokenizers.__version__), code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    # 1. files
    paths = {}
    for i in train_files + [VAL_FILE]:
        if a.smoke:
            paths[i] = Path(a.shard_dir) / Path(name(i)).name
        else:
            paths[i], h = download(i, work, log); meta['source']['file_sha256'][name(i)] = h; log(f'{name(i)} ok')

    # 2. hashes of every training document (for the exact-overlap filter on validation)
    train_hashes, n_train_docs = set(), 0
    for i in train_files:
        for batch in docs(paths[i]):
            for t in batch:
                train_hashes.add(hashlib.sha1(t.encode('utf-8')).digest()); n_train_docs += 1
    log(f'{n_train_docs} training documents, {len(train_hashes)} distinct')

    # 3. tokenizer on the first tok_docs training documents
    def tok_iter():
        n = 0
        for i in train_files:
            for batch in docs(paths[i], batch=1000):
                take = batch[:tok_docs - n]; n += len(take)
                yield take
                if n >= tok_docs:
                    return
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, min_frequency=2, special_tokens=[EOS], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.train_from_iterator(tok_iter(), trainer, length=tok_docs)
    assert tok.get_vocab_size() == VOCAB and tok.token_to_id(EOS) == 0, (tok.get_vocab_size(), tok.token_to_id(EOS))
    tok.save(str(out / 'tokenizer.json')); log(f'tokenizer trained, vocab {tok.get_vocab_size()}')

    # 4. bytes per token
    dec = byte_decoder()
    tb = np.array([1 if t == EOS else len(t) for t in (tok.id_to_token(j) for j in range(VOCAB))], dtype=np.int32)
    assert all(ch in dec for j in range(1, VOCAB) for ch in tok.id_to_token(j))
    np.save(out / 'token_bytes.npy', tb)

    def encode(batch):
        enc = tok.encode_batch(batch, add_special_tokens=False)
        ids = np.fromiter((x for e in enc for x in e.ids + [0]), dtype=np.uint16)
        return enc, ids

    # 5. validation text (first val_docs documents of the validation file that are not exact training copies)
    vd, dropped = [], 0
    for batch in docs(paths[VAL_FILE]):
        for t in batch:
            if hashlib.sha1(t.encode('utf-8')).digest() in train_hashes:
                dropped += 1; continue
            vd.append(t)
            if len(vd) == val_docs:
                break
        if len(vd) == val_docs:
            break
    enc, val_ids = encode(vd)
    for t, e in zip(vd[:500], enc[:500]):                                         # exact round trip and byte accounting
        assert tok.decode(e.ids, skip_special_tokens=False) == t, 'round trip failed'
        assert int(tb[e.ids].sum()) == len(t.encode('utf-8')), 'byte count failed'
    (out / 'val.bin').write_bytes(val_ids.tobytes())
    val_bytes = sum(len(t.encode('utf-8')) for t in vd) + len(vd)
    log(f'validation: {len(vd)} documents, {len(val_ids)} tokens, {dropped} exact training copies dropped')

    # 6. training text
    n_tok, n_bytes = 0, 0
    with open(out / 'train.bin', 'wb') as f:
        for i in train_files:
            for batch in docs(paths[i]):
                _, ids = encode(batch); f.write(ids.tobytes()); n_tok += len(ids); n_bytes += sum(len(t.encode('utf-8')) + 1 for t in batch)
            log(f'{name(i)} tokenized, {n_tok} training tokens so far')

    # 7. meta
    meta['sha256'] = {n: sha256_file(out / n) for n in ('train.bin', 'val.bin', 'token_bytes.npy', 'tokenizer.json')}
    meta.update(vocab=VOCAB, counts=dict(train_docs=n_train_docs, train_tokens=n_tok, train_bytes=n_bytes, val_docs=len(vd), val_tokens=int(len(val_ids)),
                                         val_bytes=val_bytes, val_dropped_exact_copies=dropped),
                bytes_per_token=dict(train=n_bytes / n_tok, val=val_bytes / len(val_ids)), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                seconds=round(time.time() - t0, 1))
    (out / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n', encoding='utf-8', newline='\n')
    log(json.dumps(dict(sha256=meta['sha256'], counts=meta['counts'], bytes_per_token=meta['bytes_per_token'])))


if __name__ == '__main__':
    main()
