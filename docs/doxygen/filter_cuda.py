#!/usr/bin/env python3
"""Simple filter to sanitise CUDA sources for Doxygen.

This script reads C/C++/CUDA source from stdin, strips common CUDA
qualifiers (e.g. __global__, __device__, __host__) and replaces a few
commonly-used CUDA typedefs with simple stand-ins so Doxygen's parser
can attach comments to kernels and functions reliably.

Run via Doxygen's INPUT_FILTER setting, for example in Doxyfile:

    INPUT_FILTER = "python3 docs/doxygen/filter_cuda.py"

The filter is intentionally conservative and only performs textual
rewrites that are safe for static parsing (it does not attempt to
preprocess macros beyond the common CUDA qualifiers).
"""

import sys
import re

# Doxygen may invoke the filter in two ways:
#  - pipe the file contents to the filter's stdin
#  - call the filter with the filename as an argument and *not* pipe
#    the contents
# To be robust we prefer to read the filename given on argv[1] when
# present; otherwise fall back to reading stdin. This avoids hangs when
# Doxygen does not close stdin.
if len(sys.argv) > 1:
    # Doxygen passed a filename; open and read it directly.
    try:
        with open(sys.argv[1], "r", encoding="utf-8", errors="surrogateescape") as fh:
            text = fh.read()
    except Exception:
        # Fall back to reading stdin if file open fails for any reason.
        text = sys.stdin.read()
else:
    # No filename argument — read from stdin (the common case).
    text = sys.stdin.read()
# Remove CUDA qualifiers
qualifiers = [
    r"\b__global__\b",
    r"\b__device__\b",
    r"\b__host__\b",
    r"\b__shared__\b",
    r"\b__constant__\b",
    r"\b__forceinline__\b",
    r"\b__inline__\b",
]
for q in qualifiers:
    text = re.sub(q, "", text)

# Replace common CUDA types with simple stand-ins Doxygen can parse
replacements = {
    r"\bcudaTextureObject_t\b": "unsigned long",
    r"\bTexture2DHandle\b": "unsigned long",
    r"\bTexture3DHandle\b": "unsigned long",
    r"\bfloat2\b": "struct { float x; float y; }",
    r"\bfloat3\b": "struct { float x; float y; float z; }",
}
for pattern, repl in replacements.items():
    text = re.sub(pattern, repl, text)

# Some code uses tex3D<float>(...) syntax which can confuse the parser;
# replace template tex3D<T> with a function-like call tex3D(...) for parsing
text = re.sub(r"tex3D\s*<\s*float\s*>", "tex3D", text)

# Emit cleaned source
sys.stdout.write(text)
