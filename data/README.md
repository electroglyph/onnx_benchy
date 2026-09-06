# Benchmark corpus

`fineweb-10mb.txt`: ~10 MiB of real web text, one document per line (UTF-8),
vendored from `HuggingFaceFW/fineweb`, config `sample-10BT`, split `train`
(`text` column). License: ODC-By.

Regenerate with (needs network):

```bash
.venv/bin/python scripts/fetch_fineweb.py
```

See `fineweb-10mb.meta.json` for source revision, byte/line counts, and hash.
`scripts/fetch_fineweb.py` asserts the file stays within 9–11 MB.
