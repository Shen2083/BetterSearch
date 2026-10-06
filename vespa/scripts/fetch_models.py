"""Download the model files the application package needs, on the host.

Not inside the container. Processes inside containers in this environment
cannot reach the outbound proxy and do not trust its CA, so Vespa cannot fetch
a model by url at deploy time. Everything is fetched here and referenced from
the package with path=, which is also how a production package would ship a
model it wants pinned.

answerai-colbert-small-v1 publishes a `vespa_colbert.onnx` in its own
repository, exported for exactly this embedder. We use that rather than an
export of our own, so the measurement tests Vespa's supported path instead of
testing our conversion.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

HOST = "https://huggingface.co"
FILES = [
    ("answerdotai/answerai-colbert-small-v1", "vespa_colbert.onnx", "colbert.onnx"),
    ("answerdotai/answerai-colbert-small-v1", "tokenizer.json", "colbert-tokenizer.json"),
    # bge-base-en-v1.5, so Vespa can embed the query itself rather than being
    # handed a vector from our encoder. No `prepend` is configured for it, on
    # purpose: `bettersearch.embeddings.local` adds no instruction prefix to a
    # query either, and a prefix on one side only would make the comparison
    # measure the prefix.
    # Underscores, not hyphens: Vespa derives a model's name from its filename
    # under models/ and rejects a package whose model name is not letters,
    # numbers or underscores. It only applies this to the models it imports, so
    # colbert-tokenizer.json was fine and bge-base.onnx was not.
    ("BAAI/bge-base-en-v1.5", "onnx/model.onnx", "bge_base.onnx"),
    ("BAAI/bge-base-en-v1.5", "tokenizer.json", "bge_base_tokenizer.json"),
]
DEST = Path("vespa/app/models")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for repo, remote, local in FILES:
        target = DEST / local
        if target.exists():
            print(f"  have {local} ({target.stat().st_size / 1e6:.0f} MB)")
            continue
        url = f"{HOST}/{repo}/resolve/main/{remote}"
        print(f"  fetching {repo}/{remote}")
        with urllib.request.urlopen(url) as response, target.open("wb") as handle:
            handle.write(response.read())
        print(f"  wrote {local} ({target.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
