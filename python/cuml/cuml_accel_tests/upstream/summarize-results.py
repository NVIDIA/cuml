#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize test results from a JUnit XML report file."""

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

import yaml


class QuoteTestID(str):
    """String subclass to force quoting of test IDs."""

    pass


def setup_yaml():
    """Configure YAML dumper with custom string handling."""

    def quoted_scalar(dumper, data):
        scalar_tag = "tag:yaml.org,2002:str"
        return dumper.represent_scalar(scalar_tag, data, style='"')

    yaml.add_representer(QuoteTestID, quoted_scalar)
    yaml.add_representer(
        OrderedDict,
        lambda dumper, data: dumper.represent_mapping(
            "tag:yaml.org,2002:map", data.items()
        ),
    )


def load_config(config_path):
    """Load configuration from YAML file."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize test results from a JUnit XML report file"
    )
    parser.add_argument(
        "report_file",
        type=Path,
        help="Path to the JUnit XML report file",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to config file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed failure information",
    )
    parser.add_argument(
        "-f",
        "--fail-below",
        type=float,
        help="Minimum pass rate threshold [0-100] (default: 0)",
    )
    parser.add_argument(
        "--total-fail-below",
        type=float,
        help=(
            "Minimum total pass rate threshold [0-100], including outcomes "
            "excluded by --exclude-xfail-reason (default: disabled)"
        ),
    )
    parser.add_argument(
        "--format",
        choices=["summary", "xfail_list", "traceback"],
        default="summary",
        help="Output format (default: summary)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit output to first N entries (default: no limit)",
    )
    parser.add_argument(
        "--test-id-prefix",
        type=str,
        help="Prefix to add to test IDs (e.g., 'sklearn.')",
    )
    parser.add_argument(
        "-k",
        "--filter",
        type=str,
        dest="filter_pattern",
        help="Filter tests by ID pattern (substring match, case-insensitive)",
    )
    parser.add_argument(
        "--exclude-xfail-reason",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Exclude xfails whose JUnit reason contains TEXT from the pass-rate "
            "denominator. May be specified multiple times."
        ),
    )
    args = parser.parse_args()

    # Load config if provided
    config = load_config(args.config) if args.config is not None else {}

    # Handle fail-below threshold logic:
    # 1. Use command line value if provided
    # 2. Use config value if no command line value
    # 3. Use default of 0.0 if neither is provided
    if args.fail_below is None:
        args.fail_below = config.get("threshold", {}).get("fail_below", 0.0)

    # Handle test-id-prefix logic:
    # 1. Use command line value if provided
    # 2. Use config value if no command line value
    # 3. Use empty string if neither is provided
    if args.test_id_prefix is None:
        args.test_id_prefix = config.get("test_id_prefix", "")

    return args


def validate_threshold(threshold):
    """Validate that the threshold is between 0 and 100."""
    if not 0 <= threshold <= 100:
        raise ValueError("Threshold must be between 0 and 100")


def matches_filter(test_id, pattern):
    """Check if test ID matches the filter pattern (case-insensitive substring).

    Args:
        test_id: The test ID to check
        pattern: The filter pattern (substring match, case-insensitive)

    Returns:
        True if pattern is None or test_id contains pattern
    """
    if pattern is None:
        return True
    return pattern.lower() in test_id.lower()


def matches_xfail_reason(skipped_elem, patterns):
    """Return whether an xfail reason matches any exclusion pattern."""
    message = skipped_elem.get("message", "")
    return any(pattern in message for pattern in patterns)


def get_test_results(testsuite, prefix: str = ""):
    """Extract test results from testsuite.

    Returns dict mapping test IDs to their results.
    """
    results = {}
    for testcase in testsuite.findall(".//testcase"):
        test_id = QuoteTestID(get_test_id(testcase, prefix))

        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped_elem = testcase.find("skipped")

        if failure is not None:
            msg = str(failure.get("message"))
            if "XPASS(strict)" in msg:
                status = "xpass"
            elif msg == "xfail":
                status = "xfail"
            else:
                status = "fail"
        elif error is not None:
            if "XPASS(strict)" in str(error.get("message")):
                status = "xpass"
            else:
                status = "fail"
        elif (
            skipped_elem is not None
            and skipped_elem.get("type") == "pytest.xfail"
        ):
            status = "xfail"
        else:
            status = "pass"

        results[test_id] = {
            "status": status,
        }

    return results


def format_table(rows, col_sep="  "):
    """Format a table with aligned columns.

    Args:
        rows: List of rows, where each row is a list of strings
        col_sep: String to separate columns

    Returns:
        List of formatted row strings
    """
    if not rows:
        return []

    # Calculate column widths
    num_cols = len(rows[0])
    col_widths = [0] * num_cols
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Format each row
    formatted_rows = []
    for row in rows:
        formatted_cells = []
        for i, cell in enumerate(row):
            # Right-align numeric values, left-align text
            if i > 0 and cell.replace(".", "").replace("%", "").isdigit():
                formatted_cells.append(f"{cell:>{col_widths[i]}}")
            else:
                formatted_cells.append(f"{cell:<{col_widths[i]}}")
        formatted_rows.append(col_sep.join(formatted_cells))

    return formatted_rows


def get_test_id(testcase, prefix: str = "") -> str:
    classname = testcase.get("classname", "")
    name = testcase.get("name")
    base_id = f"{classname}::{name}" if classname else name
    # Add prefix if provided and not already present
    if prefix and not base_id.startswith(prefix):
        return f"{prefix}{base_id}"
    return base_id


def format_traceback_output(
    testsuite, limit=None, prefix: str = "", filter_pattern=None
):
    """Format test results showing tracebacks of failed tests.

    Args:
        testsuite: XML testsuite element containing test results
        limit: Optional limit on number of entries to show
        prefix: Prefix to add to test IDs
        filter_pattern: Optional pattern to filter test IDs

    Returns:
        List of formatted strings containing test results and tracebacks
    """
    output = []
    output.append("Failed Tests with Tracebacks:")
    output.append("=" * 80)

    count = 0
    for testcase in testsuite.findall(".//testcase"):
        if limit is not None and count >= limit:
            output.append(f"\n... (showing first {limit} entries)")
            break

        failure = testcase.find("failure")
        error = testcase.find("error")

        if failure is not None or error is not None:
            test_id = get_test_id(testcase, prefix)

            # Apply filter
            if not matches_filter(test_id, filter_pattern):
                continue

            msg = ""
            details = ""

            if failure is not None:
                msg = failure.get("message", "")
                details = failure.text
            elif error is not None:
                msg = error.get("message", "")
                details = error.text

            if "XPASS" in msg:
                continue  # Skip xpassed tests
            elif msg == "xfail":
                continue  # Skip xfailed tests

            output.append(f'\nTest: "{test_id}"')
            output.append("-" * 80)
            if msg:
                output.append(f"Error: {msg}")
            if details:
                output.append("\nTraceback:")
                output.append(details)
            output.append("=" * 80)
            count += 1

    return output


def main():
    """Main entry point."""
    args = parse_args()
    validate_threshold(args.fail_below)
    if args.total_fail_below is not None:
        validate_threshold(args.total_fail_below)

    if not args.report_file.exists():
        print(f"Error: Report file not found: {args.report_file}")
        sys.exit(1)

    try:
        tree = ET.parse(args.report_file)
    except ET.ParseError as e:
        print(f"Error: Invalid XML file: {e}")
        sys.exit(1)

    root = tree.getroot()
    testsuite = root.find("testsuite")
    if testsuite is None:
        print("Error: No testsuite element found in XML file")
        sys.exit(1)

    # Extract test statistics
    total_tests = int(testsuite.get("tests", 0))
    time = float(testsuite.get("time", 0))

    # Count non-passing outcomes separately. Pytest records xfails as skipped
    # in JUnit XML, so count regular skips from testcase elements to avoid
    # subtracting xfails twice from the pass rate.
    regular_failures = 0
    regular_errors = 0
    regular_skipped = 0
    xfailed = 0
    excluded_xfailed = 0
    xpassed_strict = 0
    xpassed_non_strict = 0
    for testcase in testsuite.findall(".//testcase"):
        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped_elem = testcase.find("skipped")

        if failure is not None:
            msg = str(failure.get("message"))
            if "XPASS(strict)" in msg:
                xpassed_strict += 1
            elif "XPASS" in msg:
                xpassed_non_strict += 1
            elif msg == "xfail":
                xfailed += 1
            else:
                regular_failures += 1
        elif error is not None:
            msg = str(error.get("message"))
            if "XPASS(strict)" in msg:
                xpassed_strict += 1
            elif "XPASS" in msg:
                xpassed_non_strict += 1
            else:
                regular_errors += 1
        elif skipped_elem is not None:
            if skipped_elem.get("type") == "pytest.xfail":
                xfailed += 1
                if matches_xfail_reason(
                    skipped_elem, args.exclude_xfail_reason
                ):
                    excluded_xfailed += 1
            else:
                regular_skipped += 1

    # Calculate passed tests and pass rate
    passed = (
        total_tests
        - regular_failures
        - regular_errors
        - regular_skipped
        - xfailed
        - xpassed_strict
        - xpassed_non_strict
    )
    pass_rate_denominator = total_tests - excluded_xfailed
    pass_rate = (
        passed / pass_rate_denominator * 100
        if pass_rate_denominator > 0
        else 0
    )
    total_pass_rate = passed / total_tests * 100 if total_tests > 0 else 0

    if args.format == "traceback":
        output = format_traceback_output(
            testsuite, args.limit, args.test_id_prefix, args.filter_pattern
        )
        print("\n".join(output))
        return

    if args.format == "xfail_list":
        # Get test results
        test_results = get_test_results(testsuite, args.test_id_prefix)

        # Generate xfail list output
        xfail_list = []
        count = 0
        for test_id, result in test_results.items():
            # Apply filter first
            if not matches_filter(test_id, args.filter_pattern):
                continue
            if args.limit is not None and count >= args.limit:
                break
            if result["status"] in ("fail", "xfail"):
                if not xfail_list:
                    xfail_list.append(
                        OrderedDict(
                            [
                                ("reason", "Test fails with cuml.accel"),
                                ("strict", True),
                                ("tests", []),
                            ]
                        )
                    )
                xfail_list[0]["tests"].append(test_id)
                count += 1
        if xfail_list:
            xfail_list[0]["tests"].sort()
        # Print to stdout
        setup_yaml()
        print(yaml.dump(xfail_list, sort_keys=False, width=float("inf")))
        return

    # Print summary
    print("Test Summary:")
    rows = [
        ["Total Tests:", str(total_tests)],
        ["Passed:", str(passed)],
        ["Failed:", str(regular_failures)],
        ["XFailed:", str(xfailed)],
        ["Excluded XFailed:", str(excluded_xfailed)],
        ["XPassed (strict):", str(xpassed_strict)],
        ["XPassed (non-strict):", str(xpassed_non_strict)],
        ["Errors:", str(regular_errors)],
        ["Skipped:", str(regular_skipped)],
        ["Pass Rate:", f"{pass_rate:.2f}%"],
        ["Total Pass Rate:", f"{total_pass_rate:.2f}%"],
        ["Total Time:", f"{time:.2f}s"],
    ]
    for row in format_table(rows, "  "):
        print(f"  {row}")

    # List failed tests in verbose mode
    if (regular_failures + regular_errors) > 0 and args.verbose:
        print("\nFailed Tests:")
        count = 0
        for testcase in testsuite.findall(".//testcase"):
            if args.limit is not None and count >= args.limit:
                print(f"  ... (showing first {args.limit} entries)")
                break
            failure = testcase.find("failure")
            error = testcase.find("error")
            if failure is not None or error is not None:
                test_id = get_test_id(testcase, args.test_id_prefix)
                # Apply filter
                if not matches_filter(test_id, args.filter_pattern):
                    continue
                msg = ""
                if failure is not None and failure.get("message") is not None:
                    msg = failure.get("message")
                elif error is not None and error.get("message") is not None:
                    msg = error.get("message")
                if "XPASS" in msg:
                    continue  # Skip xpassed tests in failure list
                elif msg == "xfail":
                    print(f'  "{test_id}" (xfail)')
                else:
                    print(f'  "{test_id}"')
                count += 1

    # List strict xpasses in verbose mode
    if xpassed_strict > 0 and args.verbose:
        print("\nPotential Improvements (Strict XPASS):")
        count = 0
        for testcase in testsuite.findall(".//testcase"):
            if args.limit is not None and count >= args.limit:
                print(f"  ... (showing first {args.limit} entries)")
                break
            failure = testcase.find("failure")
            error = testcase.find("error")
            if failure is not None or error is not None:
                test_id = get_test_id(testcase, args.test_id_prefix)
                # Apply filter
                if not matches_filter(test_id, args.filter_pattern):
                    continue
                msg = ""
                if failure is not None and failure.get("message") is not None:
                    msg = failure.get("message")
                elif error is not None and error.get("message") is not None:
                    msg = error.get("message")
                if "XPASS(strict)" in msg:
                    print(f'  "{test_id}"')
                    count += 1

    threshold_failed = False
    if pass_rate < args.fail_below:
        print(
            f"\nError: Pass rate {pass_rate:.2f}% is below threshold "
            f"{args.fail_below}%"
        )
        threshold_failed = True
    if (
        args.total_fail_below is not None
        and total_pass_rate < args.total_fail_below
    ):
        print(
            f"\nError: Total pass rate {total_pass_rate:.2f}% is below "
            f"threshold {args.total_fail_below}%"
        )
        threshold_failed = True
    if threshold_failed:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
