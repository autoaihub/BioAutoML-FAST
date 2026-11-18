"""Small offline fixtures exercise deployment downloads without the 10 GB archive."""
import hashlib
import importlib.util
import io
import os
import pty
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("installer", ROOT / "deployment/install_datasets.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.destination = self.root / "datasets"
        self.cache = self.root / "cache"
        self.destination.mkdir()
        self.cache.mkdir()
        self.model = "dataset1_example_protein_0/runs/run_1/model/trained_model.sav"
        self.content = {"references.bib": b"reference", self.model: b"unsigned model fixture"}
        self.payload = self.archive(self.content)
        for name, value in {"ARCHIVE_BYTES": len(self.payload), "EXTRACTED_BYTES": sum(map(len, self.content.values())),
                            "MODEL_COUNT": 1, "SHA256": hashlib.sha256(self.payload).hexdigest()}.items():
            mock = patch.object(installer, name, value)
            mock.start()
            self.addCleanup(mock.stop)

    def archive(self, content):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as bundle:
            for name, value in content.items():
                member = tarfile.TarInfo(name)
                member.size = len(value)
                bundle.addfile(member, io.BytesIO(value))
        return subprocess.run(["zstd", "-q", "-c"], input=data.getvalue(), capture_output=True, check=True).stdout

    def test_install_preserves_existing_files_and_signed_models_on_repeat(self):
        reference = self.destination / "references.bib"
        reference.write_bytes(self.content["references.bib"])
        inode = reference.stat().st_ino
        with patch.object(installer, "urlopen", return_value=Response(self.payload)):
            installer.install(self.destination, self.cache)
        self.assertEqual(reference.stat().st_ino, inode)
        self.assertEqual((self.destination / self.model).read_bytes(), self.content[self.model])
        (self.destination / self.model).write_bytes(b"model signed by deployment")
        with patch.object(installer, "urlopen", side_effect=AssertionError("Unexpected download")):
            installer.install(self.destination, self.cache)
        self.assertEqual((self.destination / self.model).read_bytes(), b"model signed by deployment")

    def test_interrupted_download_resumes_with_verified_range(self):
        split = len(self.payload) // 2
        response = Response(b"")
        response.read = unittest.mock.Mock(side_effect=[self.payload[:split], OSError("connection lost")])
        with patch.object(installer, "urlopen", return_value=response), self.assertRaises(OSError):
            installer.download(self.cache)
        partial = self.cache / (installer.ARCHIVE + ".part")
        self.assertEqual(partial.read_bytes(), self.payload[:split])
        def resume(request, **kwargs):
            self.assertEqual(request.get_header("Range"), f"bytes={split}-")
            return Response(self.payload[split:], 206, {"Content-Range": f"bytes {split}-{len(self.payload)-1}/{len(self.payload)}"})
        with patch.object(installer, "urlopen", side_effect=resume):
            archive = installer.download(self.cache)
        self.assertEqual(archive.read_bytes(), self.payload)
        self.assertFalse(partial.exists())

    def test_server_ignoring_range_restarts_download(self):
        (self.cache / (installer.ARCHIVE + ".part")).write_bytes(self.payload[:10])
        with patch.object(installer, "urlopen", return_value=Response(self.payload)):
            self.assertEqual(installer.download(self.cache).read_bytes(), self.payload)

    def test_checksum_failure_does_not_install(self):
        bad = bytes([self.payload[0] ^ 1]) + self.payload[1:]
        with patch.object(installer, "urlopen", return_value=Response(bad)), self.assertRaisesRegex(ValueError, "checksum mismatch"):
            installer.install(self.destination, self.cache)
        self.assertFalse((self.destination / self.model).exists())
        self.assertFalse((self.cache / (installer.ARCHIVE + ".part")).exists())

    def test_conflict_aborts_before_any_dataset_file_is_installed(self):
        (self.destination / "references.bib").write_text("user's references")
        with patch.object(installer, "urlopen", return_value=Response(self.payload)), self.assertRaisesRegex(ValueError, "Existing file differs"):
            installer.install(self.destination, self.cache)
        self.assertFalse((self.destination / self.model).exists())
        self.assertEqual((self.destination / "references.bib").read_text(), "user's references")
        self.assertFalse((self.destination / installer.MARKER).exists())

    def test_archive_traversal_is_rejected(self):
        archive = self.cache / "malicious.tar.zst"
        archive.write_bytes(self.archive({"../outside": b"bad"}))
        with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
            installer.extract(archive, self.destination)
        self.assertFalse((self.root / "outside").exists())

    def test_existing_symlink_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.destination / "dataset1_example_protein_0").symlink_to(outside, target_is_directory=True)
        with patch.object(installer, "urlopen", return_value=Response(self.payload)), self.assertRaisesRegex(ValueError, "symlink"):
            installer.install(self.destination, self.cache)
        self.assertEqual(list(outside.iterdir()), [])

    def test_missing_installed_file_is_reported(self):
        with patch.object(installer, "urlopen", return_value=Response(self.payload)):
            installer.install(self.destination, self.cache)
        (self.destination / self.model).unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            installer.install(self.destination, self.cache)

    def test_insufficient_space_stops_before_download(self):
        with patch.object(installer.shutil, "disk_usage", return_value=SimpleNamespace(free=0)), \
                patch.object(installer, "urlopen", side_effect=AssertionError("Unexpected download")), \
                self.assertRaisesRegex(ValueError, "Insufficient disk space"):
            installer.install(self.destination, self.cache)


class LauncherTests(unittest.TestCase):
    def launch(self, answers=None, *, fail_step="install_datasets.py", state_dir=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launcher = root / "run-docker.sh"
            launcher.write_bytes((ROOT / "run-docker.sh").read_bytes())
            executable = root / "docker"
            executable.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$TEST_DOCKER_LOG"
case "$1 $2" in
    "--version ") echo 'Docker test stub' ;;
    "container inspect") echo 'run-docker.sh' ;;
    "image inspect") echo 'sha256:test-image' ;;
    "run "*)
        if [[ -n "$TEST_FAIL_STEP" && "$*" == *"$TEST_FAIL_STEP"* ]]; then exit 23; fi
        if [[ "$*" == *signing_keys.py* ]]; then echo test-key-id; fi ;;
esac
''')
            executable.chmod(0o755)
            env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}", TEST_DOCKER_LOG=str(root / "docker.log"),
                       TEST_FAIL_STEP=fail_step, MODEL_KEYS_DIR=str(root / "keys"), MODEL_SIGNING_KEY="", MODEL_TRUSTED_KEYS="",
                       MODEL_VERIFICATION_DIR=str(root / "verification"), DATASETS_DIR=str(root / "datasets"),
                       DATASETS_CACHE_DIR=str(root / "downloads"), JOBS_DIR=str(root / "jobs"), STATE_DIR=str(state_dir or root / "state"),
                       BLABLADOR_API_KEY="", MAILGUN_API_KEY="")
            if answers is None:
                result = subprocess.run(["bash", str(launcher)], env=env,
                                        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)
            else:
                master, slave = pty.openpty()
                try:
                    os.write(master, answers.encode())
                    result = subprocess.run(["bash", str(launcher)], env=env,
                                            stdin=slave, capture_output=True, text=True, timeout=10)
                finally:
                    os.close(master)
                    os.close(slave)
            commands = (root / "docker.log").read_text().splitlines()
            return result, commands

    def test_yes_installs_and_failure_preserves_running_containers(self):
        result, commands = self.launch("yes\n\n\n\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Existing containers were not stopped", result.stderr)
        install_command = next(c for c in commands if "install_datasets.py" in c)
        self.assertNotIn("signing.pem", install_command)
        self.assertNotIn("API_KEY", install_command)
        self.assertFalse(any(c.startswith(("stop ", "container rm ", "run --detach")) for c in commands))

    def test_signing_runs_offline_after_install_and_before_startup(self):
        for fail_step in ("", "deployment.sign_repository"):
            with self.subTest(fail_step=fail_step):
                result, commands = self.launch("yes\n\n\n\n", fail_step=fail_step)
                signing = next(c for c in commands if "-m deployment.sign_repository /datasets" in c)
                self.assertIn("--network none", signing)
                self.assertIn("target=/run/bioautoml/signing.pem,readonly", signing)
                self.assertNotIn("/downloads", signing)
                install = next(c for c in commands if "install_datasets.py" in c)
                self.assertLess(commands.index(install), commands.index(signing))
                starts = [c for c in commands if c.startswith("run --detach")]
                if fail_step:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(starts)
                    self.assertFalse(any(c.startswith(("stop ", "container rm ")) for c in commands))
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertLess(commands.index(signing), commands.index(starts[0]))

    def test_no_and_enter_skip_download(self):
        for answer in ("no", ""):
            with self.subTest(answer=answer):
                result, commands = self.launch(answer + "\n\n\n\n")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Download and configure", result.stderr)
                self.assertFalse(any("install_datasets.py" in c or "deployment.sign_repository" in c for c in commands))

    def test_invalid_answer_reprompts(self):
        result, commands = self.launch("maybe\nn\n\n\n\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Please answer yes or no", result.stdout)
        self.assertFalse(any("install_datasets.py" in c or "deployment.sign_repository" in c for c in commands))

    def test_sample_limit_persists_across_launches(self):
        with tempfile.TemporaryDirectory() as state:
            for answers, expected, prompt in [
                (None, "yes", None),
                ("n\n\n\n\n", "yes", "[Y/n]"),
                ("n\ninvalid\nn\n\n\n", "no", "[Y/n]"),
                (None, "no", None),
                ("n\n\n\n\n", "no", "[y/N]"),
                ("n\ny\n\n\n", "yes", "[y/N]"),
            ]:
                with self.subTest(answers=answers, expected=expected):
                    result, commands = self.launch(answers, state_dir=state)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    if prompt:
                        self.assertIn(f"Keep the 5,000-sample limit per set? {prompt}", result.stderr)
                    web = next(c for c in commands if c.startswith("run --detach"))
                    self.assertIn(f"--env BIOAUTOML_LIMIT_SAMPLES={expected} ", web)
                    self.assertEqual((Path(state) / "sample-limit").read_text().strip(), expected)

    def test_noninteractive_launch_skips_download(self):
        result, commands = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Download and configure", result.stderr)
        self.assertFalse(any("install_datasets.py" in c or "deployment.sign_repository" in c for c in commands))


if __name__ == "__main__":
    unittest.main()
