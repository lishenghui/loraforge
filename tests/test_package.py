# SPDX-License-Identifier: Apache-2.0
"""Check the installed distribution, including in environments without GPU libraries."""

import json
import subprocess
import sys
from importlib import metadata, resources

import loraforge


def test_distribution_metadata_and_package_data():
    assert loraforge.__version__ == metadata.version("loraforge")
    assert resources.files("loraforge").joinpath("py.typed").is_file()


def test_import_without_gpu_or_framework_dependencies(tmp_path):
    # A fresh, isolated interpreter avoids dependencies already imported by pytest.
    # Reject attempts, rather than only inspecting what remains in sys.modules.
    program = """
import importlib.abc
import json
import sys

class RejectGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'sglang', 'vllm'}:
            raise AssertionError(f'Unexpected dependency on import: {fullname}')

sys.meta_path.insert(0, RejectGPUImports())
import loraforge
print(json.dumps({'version': loraforge.__version__}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == {"version": loraforge.__version__}
