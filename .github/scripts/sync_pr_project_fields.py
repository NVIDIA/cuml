#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


RELEASE_PATTERN = re.compile(r"^\d+(?:\.\d+)+$")
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}
# The only Status an issue may hold and still be advanced to
# IN_PROGRESS_STATUS. Every other value, including unset, is a maintainer
# decision and is preserved.
NOT_STARTED_STATUS = "Todo"
IN_PROGRESS_STATUS = "In Progress"

QUERY = """
query(
  $owner: String!,
  $repo: String!,
  $number: Int!,
  $releaseFieldId: ID!,
  $priorityFieldId: ID!,
  $statusFieldId: ID!
) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      author {
        login
      }
      projectItems(first: 20) {
        nodes {
          ...ProjectItem
        }
      }
      closingIssuesReferences(first: 100) {
        pageInfo {
          hasNextPage
        }
        nodes {
          number
          assignees(first: 100) {
            pageInfo {
              hasNextPage
            }
            nodes {
              login
            }
          }
          repository {
            nameWithOwner
          }
          projectItems(first: 20) {
            nodes {
              ...ProjectItem
            }
          }
        }
      }
    }
  }
  releaseField: node(id: $releaseFieldId) {
    ... on ProjectV2SingleSelectField {
      options {
        id
        name
      }
    }
  }
  priorityField: node(id: $priorityFieldId) {
    ... on ProjectV2SingleSelectField {
      options {
        id
        name
      }
    }
  }
  statusField: node(id: $statusFieldId) {
    ... on ProjectV2SingleSelectField {
      options {
        id
        name
      }
    }
  }
}

fragment ProjectItem on ProjectV2Item {
  id
  project {
    id
  }
  fieldValues(first: 100) {
    nodes {
      ... on ProjectV2ItemFieldSingleSelectValue {
        name
        field {
          ... on ProjectV2SingleSelectField {
            id
          }
        }
      }
    }
  }
}
"""

UPDATE_MUTATION = """
mutation(
  $projectId: ID!,
  $itemId: ID!,
  $fieldId: ID!,
  $optionId: String!
) {
  updateProjectV2ItemFieldValue(
    input: {
      projectId: $projectId
      itemId: $itemId
      fieldId: $fieldId
      value: {singleSelectOptionId: $optionId}
    }
  ) {
    projectV2Item {
      id
    }
  }
}
"""


@dataclass(frozen=True)
class IssueFields:
    name: str
    project_item_id: str | None
    release: str | None
    priority: str | None
    status: str | None
    assignees: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr-number", required=True, type=int)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--release-field-id", required=True)
    parser.add_argument("--priority-field-id", required=True)
    parser.add_argument("--status-field-id", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the selected values without updating the project item.",
    )
    return parser.parse_args()


def release_key(release: str) -> tuple[int, ...] | None:
    if not RELEASE_PATTERN.fullmatch(release.strip()):
        return None
    return tuple(int(part) for part in release.split("."))


def choose_release(values: list[str | None]) -> str | None:
    releases = sorted({value for value in values if value})
    if not releases:
        return None
    if len(releases) == 1:
        return releases[0]

    numeric = [(release_key(value), value) for value in releases]
    if all(key is not None for key, _ in numeric):
        return min(numeric, key=lambda pair: pair[0])[1]  # type: ignore[arg-type]

    raise ValueError(
        "Cannot order conflicting non-numeric Release values: "
        + ", ".join(releases)
    )


def choose_priority(values: list[str | None]) -> str | None:
    priorities = {value for value in values if value}
    if not priorities:
        return None
    unknown = sorted(priorities - PRIORITY_ORDER.keys())
    if unknown:
        raise ValueError(
            "Unrecognized Priority value(s): " + ", ".join(unknown)
        )
    return min(priorities, key=PRIORITY_ORDER.__getitem__)


def graphql(
    token: str, query: str, variables: dict[str, Any]
) -> dict[str, Any]:
    request = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"GitHub GraphQL request failed: {exc.code} {body}"
        ) from exc

    if payload.get("errors"):
        raise RuntimeError(
            f"GitHub GraphQL returned errors: {json.dumps(payload['errors'])}"
        )
    return payload["data"]


def project_item(
    node: dict[str, Any], project_id: str
) -> dict[str, Any] | None:
    for item in node.get("projectItems", {}).get("nodes", []):
        if item.get("project", {}).get("id") == project_id:
            return item
    return None


def field_value(item: dict[str, Any] | None, field_id: str) -> str | None:
    if item is None:
        return None
    for value in item.get("fieldValues", {}).get("nodes", []):
        if value.get("field", {}).get("id") == field_id:
            return value.get("name")
    return None


def extract(
    data: dict[str, Any],
    project_id: str,
    release_field_id: str,
    priority_field_id: str,
    status_field_id: str,
) -> tuple[
    str,
    dict[str, Any],
    str | None,
    str | None,
    list[IssueFields],
]:
    pr = data.get("repository", {}).get("pullRequest")
    if pr is None:
        raise RuntimeError("GraphQL response did not include the pull request")

    author = pr.get("author", {}).get("login")
    if not author:
        raise RuntimeError("GraphQL response did not include the PR author")

    closing_issues = pr.get("closingIssuesReferences", {})
    if closing_issues.get("pageInfo", {}).get("hasNextPage"):
        raise RuntimeError(
            "The PR has more than 100 closing issue references; refusing to "
            "synchronize incomplete project data."
        )

    pr_item = project_item(pr, project_id)
    if pr_item is None:
        raise RuntimeError("The PR is not present in the configured project")

    issues = []
    for issue in closing_issues.get("nodes", []):
        assignees = issue.get("assignees", {})
        if assignees.get("pageInfo", {}).get("hasNextPage"):
            raise RuntimeError(
                f"Issue #{issue['number']} has more than 100 assignees; "
                "refusing to use incomplete assignment data."
            )
        item = project_item(issue, project_id)
        issues.append(
            IssueFields(
                name=(
                    f"{issue.get('repository', {}).get('nameWithOwner')}"
                    f"#{issue['number']}"
                ),
                project_item_id=item.get("id") if item else None,
                release=field_value(item, release_field_id),
                priority=field_value(item, priority_field_id),
                status=field_value(item, status_field_id),
                assignees=tuple(
                    node["login"] for node in assignees.get("nodes", [])
                ),
            )
        )

    return (
        author,
        pr_item,
        field_value(pr_item, release_field_id),
        field_value(pr_item, priority_field_id),
        issues,
    )


def option_id(data: dict[str, Any], field: str, value: str) -> str:
    for option in data.get(field, {}).get("options", []):
        if option.get("name") == value:
            return option["id"]
    raise RuntimeError(f"{field} does not define an option named {value!r}")


def update_field(
    token: str,
    project_id: str,
    item_id: str,
    field_id: str,
    current: str | None,
    desired: str,
    desired_option_id: str,
    dry_run: bool,
) -> None:
    if current == desired:
        return
    if dry_run:
        return

    variables = {
        "projectId": project_id,
        "itemId": item_id,
        "fieldId": field_id,
    }
    graphql(
        token,
        UPDATE_MUTATION,
        {**variables, "optionId": desired_option_id},
    )


def set_issue_in_progress(
    token: str,
    project_id: str,
    issue: IssueFields,
    status_field_id: str,
    in_progress_option_id: str,
    dry_run: bool,
) -> None:
    if issue.status != NOT_STARTED_STATUS:
        return
    if issue.project_item_id is None:
        print(
            f"warning: {issue.name} is not in the configured project; "
            "cannot set its Status",
            file=sys.stderr,
        )
        return
    update_field(
        token,
        project_id,
        issue.project_item_id,
        status_field_id,
        issue.status,
        IN_PROGRESS_STATUS,
        in_progress_option_id,
        dry_run,
    )


def main() -> int:
    args = parse_args()
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN is not set")

    owner, name = args.repo.split("/", 1)
    variables = {
        "owner": owner,
        "repo": name,
        "number": args.pr_number,
        "releaseFieldId": args.release_field_id,
        "priorityFieldId": args.priority_field_id,
        "statusFieldId": args.status_field_id,
    }
    data = graphql(token, QUERY, variables)
    author, pr_item, current_release, current_priority, issues = extract(
        data,
        args.project_id,
        args.release_field_id,
        args.priority_field_id,
        args.status_field_id,
    )

    failed = False
    # Planning fields and issue actions are independent. Keep processing after
    # an API or selection failure, but report a failing exit status at the end.
    for label, field, field_id, current, values, choose in (
        (
            "Release",
            "releaseField",
            args.release_field_id,
            current_release,
            [issue.release for issue in issues],
            choose_release,
        ),
        (
            "Priority",
            "priorityField",
            args.priority_field_id,
            current_priority,
            [issue.priority for issue in issues],
            choose_priority,
        ),
    ):
        try:
            desired = choose(values)
            if desired is None:
                # Preserve the PR field when no associated issue supplies it.
                print(f"{label}: {current or 'unset'} (no value to inherit)")
                continue
            print(f"{label}: {current or 'unset'} -> {desired}")
            update_field(
                token,
                args.project_id,
                pr_item["id"],
                field_id,
                current,
                desired,
                option_id(data, field, desired),
                args.dry_run,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            failed = True
            print(f"error: Could not sync PR {label}: {exc}", file=sys.stderr)

    print(f"PR author: {author}")
    print("Associated issues:")
    if issues:
        for issue in issues:
            print(
                f"- {issue.name}: Release={issue.release or 'unset'}, "
                f"Priority={issue.priority or 'unset'}, "
                f"Status={issue.status or 'unset'}, "
                f"Assignees={','.join(issue.assignees) or 'none'}"
            )
            try:
                if (
                    author in issue.assignees
                    and issue.status == NOT_STARTED_STATUS
                ):
                    set_issue_in_progress(
                        token,
                        args.project_id,
                        issue,
                        args.status_field_id,
                        option_id(data, "statusField", IN_PROGRESS_STATUS),
                        args.dry_run,
                    )
            except (RuntimeError, ValueError, OSError) as exc:
                failed = True
                print(
                    f"error: Could not update {issue.name}: {exc}",
                    file=sys.stderr,
                )
    else:
        print("- none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
