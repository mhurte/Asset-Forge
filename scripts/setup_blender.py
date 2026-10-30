#!/usr/bin/env python3
"""Optional pinned Blender download and CPU import probe."""
import argparse
import os
from pathlib import Path
from setup_level0 import SOURCES, download, extract, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-root', type=Path, required=True)
    a = p.parse_args()
    main_env = a.env_root.resolve()
    tools = main_env.parent / 'assetforge-tools'
    cache = tools / 'downloads'
    cache.mkdir(parents=True, exist_ok=True)
    downloads = {}
    blender = tools / "blender"
    if not (blender / "blender").is_file():
        version = SOURCES["blender_version"]
        name = f"blender-{version}-linux-x64.tar.xz"
        base = "https://download.blender.org/release/Blender4.5/"
        archive = cache / name
        actual = download(base + name, archive)
        checksum = cache / ("blender-" + version + ".sha256")
        download(base + checksum.name, checksum)
        expected = [line.split()[0] for line in checksum.read_text().splitlines() if line.split()[-1].lstrip("*") == name]
        if expected != [actual]:
            raise ValueError("Blender checksum mismatch")
        downloads[name] = actual
        extract(archive, blender, strip=1)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(main_env / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    run([blender / "blender", "--background", "--factory-startup", "--python-exit-code", "1", "--python-expr",
         "import bpy; print(bpy.app.version_string)"], env=env)
    print('Optional Blender ready:', blender / 'blender', flush=True)


if __name__ == '__main__':
    main()
