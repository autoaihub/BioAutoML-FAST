"""Signing must support XGBoost's NaN metadata without changing predictions."""
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bioautoml import model_security
from bioautoml.model_artifacts import load_model, save_model, sign_web_model


class MetadataSigningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name)
        self.path = directory / 'model.sav'
        key = Ed25519PrivateKey.generate()
        public = base64.b64encode(key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
        trust = directory / 'trust.json'
        trust.write_text(json.dumps({'keys': {'test': public}, 'revoked': []}))
        for context in (patch.object(model_security, '_signing_key', key),
                        patch.object(model_security, '_key_id', 'test'),
                        patch.dict(os.environ, {'BIOAUTOML_TRUSTED_KEYS': str(trust)})):
            context.start()
            self.addCleanup(context.stop)
        self.frame = pd.DataFrame({'x': [0., 1., np.nan, 3.]})
        self.estimator = XGBRegressor(n_estimators=2, max_depth=1, n_jobs=1)
        self.estimator.fit(self.frame, [0., 1., 2., 3.])
        save_model({'clf': self.estimator, 'train': self.frame,
                    'column_train': self.frame.columns}, self.path)

    def verify_roundtrip(self):
        self.path.chmod(0o640)
        with zipfile.ZipFile(self.path) as z:
            before = {name: z.read(name) for name in z.namelist() if name.endswith('.joblib')}
        sign_web_model(self.path)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o640)
        model = load_model(self.path)
        self.assertTrue(model.verified)
        self.assertEqual(model.info['params']['missing'], 'NaN')
        self.assertTrue(np.isnan(model['clf'].get_params()['missing']))
        np.testing.assert_array_equal(model['clf'].predict(self.frame), self.estimator.predict(self.frame))
        with zipfile.ZipFile(self.path) as z:
            for name, content in before.items():
                self.assertEqual(z.read(name), content)
            json.loads(z.read('manifest.json'), parse_constant=lambda x: self.fail(f'Nonfinite JSON: {x}'))

    def test_repository_signing_requires_pinned_manifest_and_payload(self):
        with zipfile.ZipFile(self.path) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        manifest = json.loads(members['manifest.json'])
        pin = hashlib.sha256(model_security.canonical_manifest(manifest)).hexdigest()
        with self.assertRaisesRegex(ValueError, 'pinned Zenodo release'):
            sign_web_model(self.path, expected_manifest_sha256='0' * 64)
        members['prediction.joblib'] += b'changed'
        with zipfile.ZipFile(self.path, 'w') as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        with self.assertRaisesRegex(ValueError, 'changed before signing'):
            sign_web_model(self.path, expected_manifest_sha256=pin)
        with zipfile.ZipFile(self.path) as archive:
            self.assertNotIn('signature.json', archive.namelist())

    def test_new_model_metadata_is_signable(self):
        with zipfile.ZipFile(self.path) as z:
            info = json.loads(z.read('manifest.json'), parse_constant=lambda x: self.fail(x))['info']
            self.assertEqual(info['params']['missing'], 'NaN')
        self.verify_roundtrip()

    def test_existing_unsigned_nan_metadata_is_signable(self):
        with zipfile.ZipFile(self.path) as z:
            members = {name: z.read(name) for name in z.namelist()}
        manifest = json.loads(members['manifest.json'])
        manifest['info']['params']['missing'] = float('nan')
        members['manifest.json'] = json.dumps(manifest).encode()
        with zipfile.ZipFile(self.path, 'w', compression=zipfile.ZIP_STORED) as z:
            for name, content in members.items():
                z.writestr(name, content)
        self.verify_roundtrip()


if __name__ == '__main__':
    unittest.main()
