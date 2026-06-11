#!/usr/bin/env python3
"""
Download a GGUF model into ./volume/models so the `llm` (llama.cpp) service can serve it.

Writes into the HuggingFace cache layout (volume/models/hub/models--<org>--<repo>/...),
which matches the container's HF_HOME=/workspace/.cache/huggingface. Runs on the HOST
as your user, so there is no container uid permission issue.

Setup (once):
    pip install -U "huggingface_hub[hf_transfer]"

Usage:
    python3 download_model.py unsloth/Qwen3.6-27B-MTP-GGUF -i "*UD-Q4_K_XL*"
    python3 download_model.py unsloth/gemma-4-12b-it-GGUF  -i "*UD-Q4_K_XL*" "mmproj*"   # gated: needs HF_TOKEN
    HF_TOKEN=hf_xxx python3 download_model.py <repo> -i "*Q4_K_M*"

After download, set MODEL_HUB / MODEL_GLOB in .env (the script prints the exact values),
then: docker compose up -d llm   (or: docker compose restart llm)
"""
import argparse
import os
import sys

# Resolve ./volume/models relative to this script, regardless of CWD.
ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "volume", "models")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download a GGUF model into ./volume/models")
    ap.add_argument("repo", help="HF repo id, e.g. unsloth/Qwen3.6-27B-MTP-GGUF")
    ap.add_argument(
        "-i", "--include", nargs="*", default=["*Q4_K_XL*"],
        help='filename patterns to download (default: "*Q4_K_XL*"). '
             'Add "mmproj*" for vision models.',
    )
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token for gated models (or set HF_TOKEN env)")
    ap.add_argument("--no-hf-transfer", action="store_true",
                    help="disable the fast Rust downloader (use on flaky networks)")
    args = ap.parse_args()

    if not args.no_hf_transfer:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    # Make snapshot_download write into volume/models/hub/... (matches container HF_HOME).
    os.environ["HF_HOME"] = MODELS_DIR

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print('ERROR: huggingface_hub not installed. Run:\n'
              '  pip install -U "huggingface_hub[hf_transfer]"', file=sys.stderr)
        return 1

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"[download] repo={args.repo}  include={args.include}")
    print(f"[download] into {MODELS_DIR}/hub/ ...")

    try:
        path = snapshot_download(
            repo_id=args.repo,
            allow_patterns=args.include,
            token=args.token,
        )
    except Exception as e:  # noqa: BLE001 - surface any HF error plainly
        print(f"ERROR: download failed: {e}", file=sys.stderr)
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
            print("Hint: gated model. Accept the license on HF and pass HF_TOKEN.", file=sys.stderr)
        return 1

    ggufs = [
        os.path.join(r, f)
        for r, _, fs in os.walk(path)
        for f in fs if f.endswith(".gguf")
    ]
    print("\n[done] snapshot:", path)
    for g in ggufs:
        print("  GGUF:", os.path.basename(g))

    hub_dir = "models--" + args.repo.replace("/", "--")
    print("\nSet in .env:")
    print(f"  MODEL_HUB={hub_dir}")
    if ggufs:
        # suggest a glob from the first non-mmproj gguf
        main_gguf = next((os.path.basename(g) for g in ggufs
                          if "mmproj" not in os.path.basename(g).lower()), os.path.basename(ggufs[0]))
        print(f"  MODEL_GLOB=*{main_gguf.rsplit('-', 1)[-1] if '-' in main_gguf else main_gguf}")
    print("\nThen: docker compose up -d llm   (or restart llm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
