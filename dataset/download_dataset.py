import argparse
from huggingface_hub import login, snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download dataset from HuggingFace.")
    parser.add_argument("--local_save_path", type=str, required=True, help="Local directory to save dataset")
    parser.add_argument("--hugging_face_token", type=str, required=True, help="Hugging Face access token")
    parser.add_argument("--mode", type=str, choices=["all", "reconstruction", "sourcedata"], default="all", help="Download mode: all | reconstruction | sourcedata")
    args = parser.parse_args()

    login(token=args.hugging_face_token)

    allow_patterns = None
    if args.mode == "reconstruction":
        allow_patterns = ["reconstruction/*"]
    elif args.mode == "sourcedata":
        allow_patterns = ["sourcedata/*"]

    snapshot_download(
        repo_id="mli-lab/PMoC3D",
        repo_type="dataset",
        local_dir=args.local_save_path,
        allow_patterns=allow_patterns
    )

if __name__ == "__main__":
    main()