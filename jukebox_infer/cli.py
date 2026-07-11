"""Console-script entry point (`jukebox-infer`) registered in pyproject.toml.

Used to `os.path`-hack its way to a repo-root `quick_infer.py` that was never
included in the wheel ([tool.hatch.build] only ships `jukebox_infer/**`), so
the installed CLI always failed with "Error: quick_infer.py not found." The
actual argument-parsing and generation logic now lives in
`jukebox_infer.quick_infer` (packaged, importable), so this module is just a
thin re-export -- no path hacks, no `importlib` gymnastics.

Reads: jukebox_infer.quick_infer; read by: pyproject.toml [project.scripts]
"""

from jukebox_infer.quick_infer import main

if __name__ == "__main__":
    main()
