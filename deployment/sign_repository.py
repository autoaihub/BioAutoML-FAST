"""Sign the pinned Zenodo Run 1 models before the web container starts."""
import argparse
import fcntl
import json
from pathlib import Path

from bioautoml.model_artifacts import load_model, sign_web_model
from bioautoml.model_security import initialize_web_signer
from deployment.install_datasets import MODEL_COUNT


def sign_repository(root):
    pins = json.loads(Path(__file__).with_name('run1_model_manifests.json').read_text())
    if len(pins) != MODEL_COUNT:
        raise ValueError('Incomplete list of pinned Run 1 models.')
    with (root / '.run1-install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for relative, fingerprint in pins.items():
            model = root / relative
            if model.resolve(strict=True) != model or not model.is_file():
                raise ValueError(f'Expected a regular Run 1 model file: {model}')
            sign_web_model(model, expected_manifest_sha256=fingerprint)
            load_model(model)  # Verify the signature and deployed package versions.
    print(f'All {len(pins)} Run 1 models have trusted signatures.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', type=Path)
    args = parser.parse_args()
    initialize_web_signer()
    sign_repository(args.datasets.resolve(strict=True))
