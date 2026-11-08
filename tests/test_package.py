# SPDX-License-Identifier: Apache-2.0
"""Check the installed distribution, including in environments without GPU libraries."""

import json
import subprocess
import sys
from importlib import metadata, resources

import loraforge
import loraforge.adapters
import loraforge.compression
import loraforge.integrations.sglang
import loraforge.ops
import loraforge.runtime


def test_distribution_metadata_and_package_data():
    assert loraforge.__version__ == metadata.version("loraforge")
    assert resources.files("loraforge").joinpath("py.typed").is_file()
    assert (
        resources.files("loraforge.integrations")
        .joinpath("data/mi250x_fleet_dispatch.json")
        .is_file()
    )
    native = resources.files("loraforge.ops").joinpath("csrc")
    for name in (
        "bindings.cpp",
        "tsqr_full.cu",
        "tsqr_wide.cu",
        "jacobi_eigh.cu",
        "rowexpand.cu",
    ):
        assert native.joinpath(name).is_file()


def test_import_without_gpu_or_framework_dependencies(tmp_path):
    # A fresh, isolated interpreter avoids dependencies already imported by pytest.
    # Reject attempts, rather than only inspecting what remains in sys.modules.
    program = """
import importlib.abc
import json
import sys

class RejectGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'sglang', 'vllm', 'peft', 'transformers', 'safetensors'}:
            raise AssertionError(f'Unexpected dependency on import: {fullname}')

sys.meta_path.insert(0, RejectGPUImports())
import loraforge
import loraforge.adapters
import loraforge.ops
import loraforge.runtime
import loraforge.integrations.sglang
import loraforge.compression
import loraforge.ops.rowexpand
import loraforge.runtime.retained
import loraforge.adapters.sglang
import loraforge.integrations._sglang_fleet
import loraforge.integrations._sglang_qwen35
import loraforge.annotations
import loraforge.ops.nf4
import loraforge.integrations.sglang_experimental
import loraforge.integrations._sglang_gdn_storage
import loraforge.integrations._sglang_gdn
import loraforge.integrations._sglang_nf4_linear
import loraforge.integrations._sglang_nf4_compat
import loraforge.integrations._sglang_qwen35_loader
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
