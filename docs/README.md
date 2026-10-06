# Building the documentation

## Build locally

First [build and install cuML](../BUILD.md). Generate Doxygen XML before the
Sphinx documentation because Breathe reads that XML while rendering the C++ API
pages:

```bash
./build.sh cppdocs pydocs
```

The `pydocs` target automatically generates the Doxygen XML prerequisite, so it
also works on its own. Naming both targets as above makes the prerequisite
explicit without generating it twice. Doxygen writes XML under `cpp/xml/`; it
does not produce a separately published HTML API site. The Sphinx Makefile
writes the complete documentation, including the C++ API reference, to
`docs/build/html/`:

```bash
xdg-open docs/build/html/index.html
xdg-open docs/build/html/developer_guide/cpp/api/index.html
```

CI uses the `dirhtml` builder instead, staging its version of the same Sphinx
site from `docs/_html/`.
