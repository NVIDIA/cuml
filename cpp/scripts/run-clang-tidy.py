# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

EXPECTED_VERSION = "20.1.8"
VERSION_REGEX = re.compile(r"  LLVM version ([0-9.]+)")
GPU_ARCH_REGEX = re.compile(r"sm_(\d+)")
SPACES = re.compile(r"\s+")
SEPARATOR = "-" * 16
UNSUPPORTED_CLANG_FLAGS = ("-fno-merge-constants",)


def _read_config_file(config_file):
    try:
        return tomllib.loads(Path(config_file).read_text())["tool"][
            "run-clang-tidy"
        ]
    except KeyError:
        return {}
    except tomllib.TOMLDecodeError as error:
        raise RuntimeError(
            f"Failed to read config file '{config_file}' due to syntax error: {error}"
        )


def parse_args():
    argparser = argparse.ArgumentParser("Runs clang-tidy on a project")
    argparser.add_argument(
        "-cdb",
        type=str,
        default="cpp/build/compile_commands.json",
        help="Path to cmake-generated compilation database"
        " file. It is always found inside the root of the "
        "cmake build folder. So make sure that `cmake` has "
        "been run once before running this script!",
    )
    argparser.add_argument(
        "-exe", type=str, default="clang-tidy", help="Path to clang-tidy exe"
    )
    argparser.add_argument(
        "-ignore",
        type=str,
        default="[.]cu$",
        help="Regex used to ignore files from checking",
    )
    argparser.add_argument(
        "-select",
        type=str,
        default=None,
        help="Regex used to select files for checking",
    )
    argparser.add_argument(
        "-j", type=int, default=-1, help="Number of parallel jobs to launch."
    )
    argparser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to config file (default=pyproject.toml).",
    )
    args = argparser.parse_args()

    # Read from config file.
    try:
        config = _read_config_file(args.config or "./pyproject.toml")
    except FileNotFoundError:
        if args.config:
            # only raise if config argument was explicitly used
            raise RuntimeError(f"File '{args.config}' does not exist.")
    else:
        argparser.set_defaults(**config)
        args = argparser.parse_args()

    if args.j <= 0:
        args.j = mp.cpu_count()
    args.ignore_compiled = re.compile(args.ignore) if args.ignore else None
    args.select_compiled = re.compile(args.select) if args.select else None
    ret = subprocess.check_output("%s --version" % args.exe, shell=True)
    ret = ret.decode("utf-8")
    version = VERSION_REGEX.search(ret)
    if version is None:
        raise Exception("Failed to figure out clang-tidy version!")
    version = version.group(1)
    if version != EXPECTED_VERSION:
        raise Exception(
            "clang-tidy exe must be v%s found '%s'"
            % (EXPECTED_VERSION, version)
        )
    if not os.path.exists(args.cdb):
        raise Exception("Compilation database '%s' missing" % args.cdb)
    return args


def get_all_commands(cdb):
    with open(cdb, "r") as fp:
        return json.load(fp)


def get_gpu_archs(command):
    archs = []
    for loc in range(len(command)):
        if command[loc] != "-gencode":
            continue
        arch_flag = command[loc + 1]
        match = GPU_ARCH_REGEX.search(arch_flag)
        if match is not None:
            archs.append("--cuda-gpu-arch=sm_%s" % match.group(1))
    return archs


def get_index(arr, item):
    try:
        return arr.index(item)
    except ValueError:
        return -1


def remove_item(arr, item):
    loc = get_index(arr, item)
    if loc >= 0:
        del arr[loc]
    return loc


def remove_item_plus_one(arr, item):
    loc = get_index(arr, item)
    if loc >= 0:
        del arr[loc + 1]
        del arr[loc]
    return loc


def get_clang_tidy_version():
    result = subprocess.run(
        ["clang-tidy", "--version"], capture_output=True, text=True, check=True
    )
    match = re.search(r"version ([0-9]+\.[0-9]+\.[0-9]+)", result.stdout)
    return match.group(1).strip()


def get_clang_includes(exe):
    dir = os.getenv("CONDA_PREFIX")
    if dir is None:
        ret = subprocess.check_output("which %s 2>&1" % exe, shell=True)
        ret = ret.decode("utf-8")
        dir = os.path.dirname(os.path.dirname(ret))
    headers = [
        "-I",
        os.path.join(dir, "include", "ClangHeaders"),
        # Also include main clang include directory for omp.h
        "-I",
        os.path.join(dir, "lib", "clang", get_clang_tidy_version(), "include"),
    ]
    return headers


def get_tidy_args(cmd, exe):
    command, file = cmd["command"], cmd["file"]
    is_cuda = file.endswith(".cu")
    command = re.split(SPACES, command)
    # Adjust compiler command
    if "c++" in command[0]:
        command[0] = "clang-cpp"
        command.insert(1, "-std=c++20")
    elif command[0][-2:] == "cc":
        command[0] = "clang"
    else:
        raise ValueError("Unable to identify compiler.")
    # remove compilation and output targets from the original command
    remove_item_plus_one(command, "-c")
    remove_item_plus_one(command, "-o")
    for flag in UNSUPPORTED_CLANG_FLAGS:
        while remove_item(command, flag) >= 0:
            pass
    if is_cuda:
        # replace nvcc's "-gencode ..." with clang's "--cuda-gpu-arch ..."
        archs = get_gpu_archs(command)
        command.extend(archs)
        while True:
            loc = remove_item_plus_one(command, "-gencode")
            if loc < 0:
                break
        # "-x cuda" is the right usage in clang
        loc = get_index(command, "-x")
        if loc >= 0:
            command[loc + 1] = "cuda"
        remove_item_plus_one(command, "-ccbin")
        remove_item(command, "--expt-extended-lambda")
        remove_item(command, "--diag_suppress=unrecognized_gcc_pragma")
    command.extend(get_clang_includes(exe))
    return command, is_cuda


def run_clang_tidy_command(tidy_cmd):
    cmd = " ".join(tidy_cmd)
    result = subprocess.run(
        cmd,
        check=False,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    passed = result.returncode == 0
    out = (
        ""
        if passed
        else f"CMD: {cmd}: {result.stdout.decode('utf-8').rstrip()}"
    )
    return passed, out


def run_clang_tidy(cmd, args):
    command, is_cuda = get_tidy_args(cmd, args.exe)
    tidy_cmd = [
        args.exe,
        "-header-filter='.*cuml/cpp/(src|include|bench|comms).*'",
        cmd["file"],
        "--",
    ]
    tidy_cmd.extend(command)
    all_passed = True
    out = []
    if is_cuda:
        tidy_cmd.append("--cuda-device-only")
        tidy_cmd.append(cmd["file"])
        passed, out1 = run_clang_tidy_command(tidy_cmd)
        out.append(out1)
        out.append(SEPARATOR)
        if not passed:
            all_passed = False
        tidy_cmd[-2] = "--cuda-host-only"
        passed, out1 = run_clang_tidy_command(tidy_cmd)
        if not passed:
            all_passed = False
        out.append(out1)
    else:
        tidy_cmd.append(cmd["file"])
        passed, out1 = run_clang_tidy_command(tidy_cmd)
        if not passed:
            all_passed = False
        out.append(out1)
        out += out1
    return all_passed, "".join(out), cmd["file"]


# yikes! global var :(
results = []


def collect_result(result):
    global results
    results.append(result)


def print_result(passed, stdout, file):
    status_str = "PASSED" if passed else "FAILED"
    print("%s File:%s %s %s" % (SEPARATOR, file, status_str, SEPARATOR))
    if stdout:
        print(stdout)
        print("%s File:%s ENDS %s" % (SEPARATOR, file, SEPARATOR))


def print_results():
    global results
    status = True
    for passed, stdout, file in results:
        print_result(passed, stdout, file)
        if not passed:
            status = False
    return status


def run_tidy_for_all_files(args, all_files):
    pool = None if args.j == 1 else mp.Pool(args.j)
    # actual tidy checker
    for cmd in all_files:
        # skip files that we don't want to look at
        if (
            args.ignore_compiled is not None
            and re.search(args.ignore_compiled, cmd["file"]) is not None
        ):
            continue
        if (
            args.select_compiled is not None
            and re.search(args.select_compiled, cmd["file"]) is None
        ):
            continue
        if pool is not None:
            pool.apply_async(
                run_clang_tidy, args=(cmd, args), callback=collect_result
            )
        else:
            passed, stdout, file = run_clang_tidy(cmd, args)
            collect_result((passed, stdout, file))
    if pool is not None:
        pool.close()
        pool.join()
    return print_results()


def main():
    args = parse_args()
    # Attempt to making sure that we run this script from root of repo always
    if not os.path.exists(".git"):
        raise Exception("This needs to always be run from the root of repo")
    # Check whether clang-tidy exists
    if shutil.which("clang-tidy") is None:
        print(
            "Unable to find the clang-tidy executable, is it installed?",
            file=sys.stderr,
        )
        sys.exit(1)
    all_files = get_all_commands(args.cdb)
    status = run_tidy_for_all_files(args, all_files)
    if not status:
        raise Exception("clang-tidy failed! Refer to the errors above.")


if __name__ == "__main__":
    main()
