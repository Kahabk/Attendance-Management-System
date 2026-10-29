"""Install official OpenCV Zoo models, verifying Git LFS SHA-256 checksums."""
from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / "assets" / "models"
MODELS = {
    "face_detection_yunet": "face_detection_yunet_2023mar.onnx",
    "face_recognition_sface": "face_recognition_sface_2021dec.onnx",
}


def download_models() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    # Resolve a single revision so checksum metadata and binaries cannot drift mid-download.
    import json
    request = urllib.request.Request("https://api.github.com/repos/opencv/opencv_zoo/commits/main", headers={"User-Agent": "attendance-model-setup"})
    with urllib.request.urlopen(request, timeout=60) as response:
        revision = json.load(response)["sha"]
    for folder, name in MODELS.items():
        path = f"models/{folder}/{name}"
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/opencv/opencv_zoo/{revision}/{path}", timeout=60) as response:
            pointer = response.read().decode("utf-8")
        expected_hash = next(line.split("sha256:", 1)[1] for line in pointer.splitlines() if line.startswith("oid sha256:"))
        expected_size = int(next(line.split()[1] for line in pointer.splitlines() if line.startswith("size ")))
        # Keep attribution available even when the binary is already installed.
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/opencv/opencv_zoo/{revision}/models/{folder}/LICENSE", timeout=60) as response:
            (ROOT / f"{folder}_LICENSE").write_bytes(response.read())
        target = ROOT / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_hash:
            print(f"Already verified: {name}")
            continue
        temporary = target.with_suffix(".onnx.part")
        try:
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{revision}/{path}", timeout=120) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    output.write(chunk)
            if digest.hexdigest() != expected_hash or size != expected_size:
                raise ValueError(f"Checksum or size mismatch: {name}")
            temporary.replace(target)
            print(f"Installed and verified: {name}")
        finally:
            temporary.unlink(missing_ok=True)
    print("Restart the app and enroll faces using the new backend.")


if __name__ == "__main__":
    download_models()
