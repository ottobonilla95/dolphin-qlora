"""Download a pinned Dolphin checkpoint and create a separate 4-bit MLX model."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess

from huggingface_hub import snapshot_download
from mlx_lm.convert import convert

ROOT = Path(__file__).resolve().parent
REPO = 'dphn/Dolphin3.0-Llama3.1-8B'
REVISION = 'f065677950dfc7e708d518d64cf1f5041ee007a0'
DESTINATION = ROOT / 'training-model' / 'dolphin-4bit'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if DESTINATION.exists():
        raise FileExistsError(f'Preserving existing output: {DESTINATION}')
    print(f'Downloading {REPO} at {REVISION}', flush=True)
    snapshot = snapshot_download(
        repo_id=REPO, revision=REVISION, max_workers=4,
        allow_patterns=['*.safetensors', 'model.safetensors.index.json', 'config.json',
                        'generation_config.json', 'tokenizer*.json', 'special_tokens_map.json', 'README.md'],
    )
    print(f'Download complete. Converting into {DESTINATION}', flush=True)
    convert(hf_path=snapshot, mlx_path=str(DESTINATION), quantize=True,
            q_bits=4, q_group_size=64, q_mode='affine')
    print('Conversion complete. Recording file checksums.', flush=True)
    packages = ['mlx', 'mlx-metal', 'mlx-lm', 'transformers', 'huggingface-hub', 'datasets', 'tokenizers']
    manifest = {
        'source_repo': REPO, 'source_revision': REVISION, 'source_snapshot': snapshot,
        'destination': str(DESTINATION),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'quantization': {'bits': 4, 'group_size': 64, 'mode': 'affine'},
        'python': platform.python_version(),
        'packages': {package: version(package) for package in packages},
        'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                  for p in sorted(DESTINATION.iterdir()) if p.is_file()},
        'training_started': False,
        'notes': 'MLX quantization differs from the original GGUF Q4_K_S. Evaluate this exact model before and after training.',
    }
    manifest_path = ROOT / 'training-model' / 'model-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    freeze = subprocess.run(
        ['uv', 'pip', 'freeze', '--python', str(ROOT / '.venv/bin/python')],
        check=True, capture_output=True, text=True,
    )
    (ROOT / 'requirements-lock.txt').write_text(freeze.stdout)
    print(f'Saved model: {DESTINATION}', flush=True)
    print(f'Files: {len(manifest["files"])}; bytes: {sum(p["bytes"] for p in manifest["files"].values())}', flush=True)
    print(f'Manifest: {manifest_path}', flush=True)


if __name__ == '__main__':
    main()
