from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="NexVeridian/gpt-oss-20b-4bit",
    local_dir="/Users/choemanseung/789/hft/mlx_models/gpt-oss-20b-4bit",
    local_dir_use_symlinks=False
)