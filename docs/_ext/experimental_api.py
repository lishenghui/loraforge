# SPDX-License-Identifier: Apache-2.0
"""Render runtime alpha metadata as an API admonition without executing APIs."""


def process_docstring(app, what, name, obj, options, lines):
    info = getattr(obj, "__experimental_api__", None)
    if info is None:
        return
    # Replace the annotation's plain-text footer with one structured notice.
    lines[:] = [
        line for line in lines if not line.startswith("Experimental API (alpha, since ")
    ]
    lines[:0] = [
        ".. admonition:: Experimental API (alpha)",
        "   :class: warning experimental-api",
        "",
        f"   Added in {info.since}. {info.reason}",
        "   API compatibility is not guaranteed. Review the documented validation limits.",
        "",
    ]


def setup(app):
    app.connect("autodoc-process-docstring", process_docstring, priority=600)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
