# Repository guidance

This is the maintained LoRAForge library. Keep changes scoped to the requested
migration step. Research repositories and their results are source evidence;
do not edit them as part of extracting library functionality.

- Read `docs/architecture.md`, `docs/migration.md`, and `CONTRIBUTING.md` before
  adding numerical implementations or integrations.
- Keep top-level imports free of device initialization, kernel compilation, and
  serving-framework imports.
- FlashTSQR is the default accelerated compression backend. Do not silently
  replace it with generic QR or relabel reference runs as FlashTSQR results.
- Preserve import provenance, licenses and attribution in the same change as
  each extraction. Do not copy experiment/result trees or paper submodules.
- Add only implemented modules; avoid empty frameworks and placeholder APIs.
- Run the checks in `CONTRIBUTING.md`. Validate numerical GPU changes on the
  intended allocated GPU hardware before claiming support. CPU work does not
  need a GPU allocation.
