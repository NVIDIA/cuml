# This contains code with copyright by the scikit-learn project, subject to the
# license in /thirdparty/LICENSES/LICENSE.scikit_learn
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
#

import inspect
import os
import re
import subprocess
import sys
from operator import attrgetter

orig = inspect.isfunction


# See https://opendreamkit.org/2017/06/09/CythonSphinx/
def isfunction(obj):
    orig_val = orig(obj)

    new_val = hasattr(type(obj), "__code__")

    if orig_val != new_val:
        return new_val

    return orig_val


inspect.isfunction = isfunction

REVISION_CMD = "git rev-parse --short HEAD"

source_regex = re.compile(
    r"^File: (.*?) \(starting at line ([0-9]*?)\)$", re.MULTILINE
)


def _get_git_revision():
    try:
        revision = subprocess.check_output(REVISION_CMD.split()).strip()
    except (subprocess.CalledProcessError, OSError):
        print("Failed to execute git to get revision")
        return None
    return revision.decode("utf-8")


def _linkcode_resolve(domain, info, package, url_fmt, revision):
    """Determine a link to online source for a class/method/function

    This is called by sphinx.ext.linkcode

    An example with a long-untouched module that everyone has
    >>> _linkcode_resolve('py', {'module': 'tty',
    ...                          'fullname': 'setraw'},
    ...                   package='tty',
    ...                   url_fmt='http://hg.python.org/cpython/file/'
    ...                           '{revision}/Lib/{package}/{path}#L{lineno}',
    ...                   revision='xxxx')
    'http://hg.python.org/cpython/file/xxxx/Lib/tty/tty.py#L18'
    """

    if revision is None:
        return
    if domain not in ("py", "pyx"):
        return
    if not info.get("module") or not info.get("fullname"):
        return

    class_name = info["fullname"].split(".")[0]
    module = __import__(info["module"], fromlist=[class_name])
    obj = attrgetter(info["fullname"])(module)

    # Unwrap the object to get the correct source
    # file in case that is wrapped by a decorator
    obj = inspect.unwrap(obj)

    fn: str = None
    lineno: str = None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None

    if not fn:
        # Possibly Cython code. Search docstring for source
        if obj.__doc__ is None:
            return
        m = source_regex.search(obj.__doc__)

        if m is not None:
            source_file = m.group(1)
            lineno = m.group(2)

            # fn is expected to be the absolute path.
            fn = os.path.relpath(source_file, start=package)
            print(
                "{}:{}".format(
                    os.path.abspath(os.path.join("..", "python", "cuml", fn)),
                    lineno,
                )
            )
        else:
            return
    else:
        if fn.endswith(".pyx"):
            sp_path = next(
                x for x in sys.path if re.match(".*site-packages$", x)
            )
            fn = fn.replace("/opt/conda/conda-bld/work/python/cuml", sp_path)

        if os.path.isabs(fn):
            # Convert to relative from module root
            fn = os.path.relpath(
                fn, start=os.path.dirname(__import__(package).__file__)
            )
        else:
            # Cython reports paths relative to the source root
            # (e.g. "cuml/cluster/dbscan.pyx"), not the working directory
            fn = os.path.relpath(fn, start=package)

    # Objects inherited from other packages (e.g. scikit-learn mixins)
    if fn.startswith(".."):
        return

    # Get the line number if we need it. (Can work without it)
    if lineno is None:
        try:
            lineno = inspect.getsourcelines(obj)[1]
        except Exception:
            # Can happen if its a cyfunction. See if it has `__code__`
            if hasattr(obj, "__code__"):
                lineno = obj.__code__.co_firstlineno
            else:
                lineno = ""
    return url_fmt.format(
        revision=revision, package=package, path=fn, lineno=lineno
    )


def make_linkcode_resolve(package, url_fmt):
    """Returns a linkcode_resolve function for the given URL format

    revision is a git commit reference (hash or name)

    package is the name of the root module of the package

    url_fmt is along the lines of ('https://github.com/USER/PROJECT/'
                                   'blob/{revision}/{package}/'
                                   '{path}#L{lineno}')
    """
    revision = _get_git_revision()

    # Using a partial function doesn't work with sphinx.ext.linkcode.
    def linkcode_resolve(domain, info):
        return _linkcode_resolve(domain, info, package, url_fmt, revision)

    return linkcode_resolve
