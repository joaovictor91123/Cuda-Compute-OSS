"""PR evaluation gate chain -- oldest-PR-first, adapted from sparkinfer's
``eval/pr_eval_bot.py`` (see docs/sn74-emission-strategy.md,
docs/testing-strategy.md).

Gate chain per open PR, in order:
    draft skip -> blocked-contributor check -> protected-path check
    -> copycat check (eval.copycat_guard) -> PR-lane classification
    -> fix/docs lane: CPU-only review
    -> feat/strategy lane: scorecard check -> GPU queue

The DECISION (:func:`process_pr`) is a pure function of already-fetched data
-- no GitHub I/O happens inside it, so the whole gate chain is unit-testable
with plain fixtures (see eval/tests/test_pr_bot.py). :class:`GitHubClient` is
the only place that talks to ``gh``; :func:`run_once` wires the two together
and is the only place writes (comment/label/close) happen, and only when
``dry_run=False``.

Phase 1 keeps write-back OFF by default -- there is no live bot identity or
GPU runner yet (Phase 2/3). The write path below is implemented and tested
now so it's ready to flip on deliberately later, not exercised against the
real repo by default.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import copycat_guard
from .github_client import GitHubClient, PRInfo

REPO_DEFAULT = "zeokin/Cuda-Compute-OSS"
BLOCKED_CONTRIBUTORS_PATH = ".github/blocked-contributors.txt"
IDEMPOTENCY_MARKER = "<!-- cco-eval:{sha} -->"
NEEDS_SCORECARD_MARKER = "<!-- cco-needs-scorecard:{sha} -->"
NEEDS_PR_KIND_MARKER = "<!-- cco-needs-pr-kind:{sha} -->"
MERGE_CONFLICT_MARKER_RE = re.compile(
    r"<!--\s*cco-merge-conflict:(?P<sha>[0-9A-Za-z._-]+):(?P<ts>[^ >]+)\s*-->"
)
BLOCKING_CHANGE_MARKER_RE = re.compile(
    r"<!--\s*cco-blocking-change:"
    r"(?P<reason>[0-9A-Za-z._-]+):(?P<sha>[0-9A-Za-z._-]+):(?P<ts>[^ >]+)\s*-->"
)
RESULT_MARKER_PREFIX = "<!-- cco-result:{pr}:"
GPU_QUEUE_LABEL = "status:queued-gpu"
GPU_QUEUE_READY_ACTIONS = frozenset({"eval_pending"})
READY_NON_GPU_LABEL = "status:ready-non-gpu"
NEEDS_PR_KIND_LABEL = "status:needs-pr-kind"
PROTECTED_PATH_LABEL = "status:protected-path"
MAX_OPEN_PRS_PER_AUTHOR = 2
MERGE_CONFLICT_GRACE = timedelta(hours=12)
BLOCKING_CHANGE_GRACE = timedelta(hours=12)
MAINTAINER_HOLD_LABELS = frozenset({
    "status:maintainer-review",
    "status:hold",
    "status:manual-review",
    "hold",
    "do-not-close",
})
PROTECTED_PATH_PREFIXES = ("eval/", "docs/", ".github/", "dashboard/")
PROTECTED_PATH_EXACT = frozenset()
FEATURE_KIND_LABELS = frozenset({"type:feature", "type:strategy", "type:enhancement"})
NON_GPU_KIND_LABELS = frozenset({"type:bug", "type:docs"})
DOCS_ONLY_SUFFIXES = (".md", ".rst", ".txt")
DOCS_ONLY_EXACT = frozenset({"LICENSE"})

# Matches .github/workflows/labeler.yml's existing status:needs-scorecard
# detector exactly (kept as one Python regex so the two never drift): treat
# the PR as missing its scorecard unless the body carries an actual filled-in
# metric, not just an empty template or a checked-but-unsubstantiated box.
# Phase 2's real runner is the actual authority on whether a PR improved
# anything -- this is only a courtesy pre-filter.
SCORECARD_RE = re.compile(
    r"accuracy[^\n]*[0-9]|latency[^\n]*[0-9]|RESULT_JSON", re.IGNORECASE
)
FIX_TITLE_RE = re.compile(r"^\s*(fix|bug)(\([^)]+\))?:|\[\s*(fix|bug)\s*\]", re.IGNORECASE)
FEAT_TITLE_RE = re.compile(
    r"^\s*(feat|feature|strategy)(\([^)]+\))?:|\[\s*(feat|feature|strategy)\s*\]",
    re.IGNORECASE,
)
FIX_BODY_RE = re.compile(
    r"^\s*-\s*\[[xX]\]\s*(fix|bug)\b|pr kind\s*:\s*(fix|bug)\b",
    re.IGNORECASE | re.MULTILINE,
)
FEAT_BODY_RE = re.compile(
    r"^\s*-\s*\[[xX]\]\s*(feat|feature|strategy)\b|pr kind\s*:\s*(feat|feature|strategy)\b",
    re.IGNORECASE | re.MULTILINE,
)
STRATEGY_HINT_RE = re.compile(
    r"register_transform|class\s+\w+\(Transform\)|\btransform\b|\bstrategy\b",
    re.IGNORECASE,
)
CODING_AGENT_COAUTHOR_RE = re.compile(
    r"(?im)^co-authored-by:\s*.*"
    r"(cursor|codex|claude|copilot|openai|anthropic|aider|windsurf|devin|"
    r"codeium|tabnine|qodo|amazon\s*q|coding[- ]?agent|ai[- ]?agent)",
)


@dataclass
class GateOutcome:
    """The bot's decision for one PR."""
    pr: int
    action: str  # see process_pr's docstring for the full set of values
    detail: str = ""
    label: str | None = None
    kind: str | None = None


def load_blocked_contributors(path: str = BLOCKED_CONTRIBUTORS_PATH) -> frozenset:
    p = Path(path)
    if not p.exists():
        return frozenset()
    return frozenset(
        line.strip() for line in p.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def already_queued(comments: list, head_sha: str) -> bool:
    marker = IDEMPOTENCY_MARKER.format(sha=head_sha)
    return any(marker in c for c in comments)


def already_notified(comments: list, marker: str, head_sha: str) -> bool:
    tagged = marker.format(sha=head_sha)
    return any(tagged in c for c in comments)


def already_evaluated(pr_number: int, comments: list) -> bool:
    marker = RESULT_MARKER_PREFIX.format(pr=pr_number)
    return any(marker in c for c in comments)


def _parse_iso8601(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def merge_conflict_comment_time(comments: list, head_sha: str) -> datetime | None:
    latest = None
    for body in comments:
        for match in MERGE_CONFLICT_MARKER_RE.finditer(body or ""):
            if match.group("sha") != head_sha:
                continue
            try:
                when = _parse_iso8601(match.group("ts"))
            except ValueError:
                continue
            if latest is None or when > latest:
                latest = when
    return latest


def blocking_change_comment_time(
    comments: list,
    head_sha: str,
    reason: str | None = None,
) -> datetime | None:
    """Latest bot blocking-change request for this exact head SHA."""
    latest = None
    for body in comments:
        for match in BLOCKING_CHANGE_MARKER_RE.finditer(body or ""):
            if match.group("sha") != head_sha:
                continue
            if reason is not None and match.group("reason") != reason:
                continue
            try:
                when = _parse_iso8601(match.group("ts"))
            except ValueError:
                continue
            if latest is None or when > latest:
                latest = when
    return latest


def has_maintainer_hold(pr: PRInfo) -> bool:
    """True when a maintainer label explicitly disables stale auto-close."""
    labels = {label.lower() for label in pr.labels}
    return bool(labels & MAINTAINER_HOLD_LABELS)


def _blocking_request_outcome(
    pr: PRInfo,
    comments: list,
    *,
    reason: str,
    active_action: str,
    active_detail: str,
    label: str | None,
    kind: str | None = None,
    close_detail: str,
    now: datetime,
) -> GateOutcome:
    """Return the active blocking request or stale-close outcome.

    The timer is keyed to the PR's current head SHA. Any new commit changes the
    SHA, so the next bot run posts a fresh request and restarts the 12-hour
    window without any external state.
    """
    warned_at = blocking_change_comment_time(comments, pr.head_sha, reason)
    if warned_at is not None and not has_maintainer_hold(pr):
        if now - warned_at >= BLOCKING_CHANGE_GRACE:
            return GateOutcome(
                pr.number,
                "close_stale_blocking_request",
                detail=close_detail,
                kind=kind,
            )
    return GateOutcome(pr.number, active_action, detail=active_detail, label=label, kind=kind)


def has_merge_conflict(pr: PRInfo) -> bool:
    status = (pr.merge_state_status or "").upper()
    mergeable = (pr.mergeable or "").upper()
    return status == "DIRTY" or mergeable == "CONFLICTING"


def has_scorecard(body: str) -> bool:
    return bool(SCORECARD_RE.search(body or ""))


def has_coding_agent_coauthor(commit_messages: str) -> bool:
    """Return True when commit footers co-author a coding agent.

    Human co-authors are fine. The rule is specifically against crediting
    coding-agent identities such as Cursor, Codex, Claude, Copilot, etc.
    """
    return bool(CODING_AGENT_COAUTHOR_RE.search(commit_messages or ""))


def changed_files(diff_text: str) -> frozenset[str]:
    files = set()
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            if path and path != "/dev/null":
                files.add(path)
    return frozenset(files)


def protected_paths(diff_text: str) -> tuple[str, ...]:
    hits = []
    for path in sorted(changed_files(diff_text)):
        if path in PROTECTED_PATH_EXACT or path.startswith(PROTECTED_PATH_PREFIXES):
            hits.append(path)
    return tuple(hits)


def _docs_only(files: frozenset[str]) -> bool:
    if not files:
        return False
    for path in files:
        if path in DOCS_ONLY_EXACT:
            continue
        if path.endswith(DOCS_ONLY_SUFFIXES):
            continue
        return False
    return True


def _tests_examples_only(files: frozenset[str]) -> bool:
    if not files:
        return False
    allowed_prefixes = ("tests/", "examples/")
    for path in files:
        if path.startswith(allowed_prefixes):
            continue
        if "/tests/" in path or path.endswith(".md"):
            continue
        if path.startswith("strategy/") and "/tests/" in path:
            continue
        return False
    return True


def classify_pr(pr: PRInfo, diff_text: str) -> str:
    """Classify a PR lane.

    Returns one of:
      - ``feat``: score-bearing feature / strategy PR, requires GPU evidence
      - ``fix``: bug fix / maintenance PR, CPU-safe CI only
      - ``docs``: documentation-only / explanatory PR, CPU-safe CI only
      - ``unknown``: contributor must declare the lane explicitly
    """
    labels = {label.lower() for label in pr.labels}
    has_feature_label = bool(labels & FEATURE_KIND_LABELS)
    has_non_gpu_label = bool(labels & NON_GPU_KIND_LABELS)
    explicit_fix = bool(FIX_TITLE_RE.search(pr.title or "") or FIX_BODY_RE.search(pr.body or ""))
    explicit_feat = bool(
        FEAT_TITLE_RE.search(pr.title or "")
        or FEAT_BODY_RE.search(pr.body or "")
    )
    files = changed_files(diff_text)

    if _docs_only(files):
        return "docs"
    if explicit_fix and explicit_feat:
        return "unknown"
    if has_feature_label and has_non_gpu_label:
        return "unknown"
    if explicit_feat or has_feature_label:
        return "feat"
    if explicit_fix or has_non_gpu_label:
        return "fix"
    if has_scorecard(pr.body):
        return "feat"
    if STRATEGY_HINT_RE.search(f"{pr.title}\n{pr.body}"):
        return "feat"
    if _tests_examples_only(files):
        return "fix"
    return "unknown"


def excess_open_prs(prs: list[PRInfo], limit: int = MAX_OPEN_PRS_PER_AUTHOR) -> frozenset[int]:
    """Return PR numbers that exceed the per-author open-PR cap.

    The bot keeps the oldest ``limit`` open PRs per author and closes newer
    overflow PRs. PR number order matches GitHub creation order, so a miner
    cannot keep opening fresh PRs while older ones are still awaiting review.
    """
    if limit <= 0:
        return frozenset()

    by_author: dict[str, list[PRInfo]] = {}
    for pr in prs:
        by_author.setdefault(pr.author, []).append(pr)

    overflow = set()
    for author_prs in by_author.values():
        ranked = sorted(author_prs, key=lambda pr: pr.number)
        for pr in ranked[limit:]:
            overflow.add(pr.number)
    return frozenset(overflow)


def process_pr(
    pr: PRInfo,
    diff_text: str,
    comments: list,
    blocked: frozenset,
    originals: list,
    excess_pr_numbers: frozenset[int] = frozenset(),
    commit_messages: str = "",
    now: datetime | None = None,
    run_eval=None,
) -> GateOutcome:
    """Decide the gate-chain outcome for one PR. Pure: takes already-fetched
    data, performs no GitHub I/O, so it's fully unit-testable.

    originals : list of (author, copycat_guard.Fingerprint) for every earlier
                PR, oldest first. The candidate's own earlier PRs are
                filtered out here (self-resubmission is not copying).
    run_eval  : callable(pr) -> dict, or None (Phase 1's stub -- gate chain
                passed but there is no GPU runner wired up yet).

    action values: skip_draft, close_blocked, close_excess_open_pr,
    needs_merge_conflict_resolution, close_stale_merge_conflict,
    close_coding_agent_coauthor, close_stale_blocking_request, copycat_block,
    copycat_warn, already_evaluated, needs_pr_kind, needs_scorecard,
    non_gpu_review, eval_pending, evaluated.
    """
    now = now or datetime.now(timezone.utc)

    if pr.is_draft:
        return GateOutcome(pr.number, "skip_draft")

    if pr.author in blocked:
        return GateOutcome(pr.number, "close_blocked",
                           detail=f"{pr.author} is on the blocked-contributors list")

    if pr.number in excess_pr_numbers:
        return GateOutcome(
            pr.number,
            "close_excess_open_pr",
            detail=f"{pr.author} already has more than {MAX_OPEN_PRS_PER_AUTHOR} open PRs; "
                   "closing this newer PR and keeping only the two oldest open ones.",
        )

    if has_coding_agent_coauthor(commit_messages):
        return GateOutcome(
            pr.number,
            "close_coding_agent_coauthor",
            detail="Commit history contains a Co-authored-by footer for a coding agent "
                   "(for example Cursor, Codex, Claude, Copilot, or similar). "
                   "CCO does not accept coding-agent co-author footers.",
        )

    if has_merge_conflict(pr):
        warned_at = merge_conflict_comment_time(comments, pr.head_sha)
        if warned_at is None:
            return GateOutcome(
                pr.number,
                "needs_merge_conflict_resolution",
                detail="This PR has merge conflicts with the base branch. Resolve them within "
                       "12 hours or the bot will close the PR automatically.",
            )
        if now - warned_at >= MERGE_CONFLICT_GRACE:
            return GateOutcome(
                pr.number,
                "close_stale_merge_conflict",
                detail="Closing this PR because the reported merge conflict was not resolved "
                       "within 12 hours of the bot reminder.",
            )
        return GateOutcome(
            pr.number,
            "needs_merge_conflict_resolution",
            detail="This PR still has merge conflicts with the base branch. Resolve them within "
                   "12 hours of the bot reminder to keep it open.",
        )

    protected = protected_paths(diff_text)
    if protected:
        detail = "PR touches maintainer-owned files: " + ", ".join(protected)
        return _blocking_request_outcome(
            pr,
            comments,
            reason="protected-path",
            active_action="protected_path",
            active_detail=detail,
            label=PROTECTED_PATH_LABEL,
            close_detail=(
                "Closing this PR because the requested protected-path split was not "
                "fixed within 12 hours of the bot reminder. Open a fresh PR after "
                "removing maintainer-owned files."
            ),
            now=now,
        )

    kind = classify_pr(pr, diff_text)
    fp = copycat_guard.fingerprint(diff_text)
    others = [(a, f) for a, f in originals if a != pr.author]
    matched_author, verdict = copycat_guard.worst_verdict(fp, others)
    if verdict.tier == "block":
        return GateOutcome(pr.number, "copycat_block",
                           detail=f"matches an earlier PR by {matched_author}: {verdict.reason}",
                           label="copycat", kind=kind)
    if verdict.tier == "warn":
        return GateOutcome(pr.number, "copycat_warn",
                           detail=f"matches an earlier PR by {matched_author}: {verdict.reason}",
                           label="copycat-warn", kind=kind)

    if kind == "unknown":
        return _blocking_request_outcome(
            pr,
            comments,
            reason="pr-kind",
            active_action="needs_pr_kind",
            active_detail=(
                "Declare this PR as either fix/bug or feat/strategy in the title "
                "or PR template so automation can route it correctly."
            ),
            label=NEEDS_PR_KIND_LABEL,
            kind=kind,
            close_detail=(
                "Closing this PR because the requested PR-kind declaration was not "
                "fixed within 12 hours of the bot reminder. Open a fresh PR with a "
                "clear fix/bug or feat/strategy lane."
            ),
            now=now,
        )

    if already_evaluated(pr.number, comments):
        return GateOutcome(pr.number, "already_evaluated", kind=kind)

    if kind in {"fix", "docs"}:
        return GateOutcome(
            pr.number,
            "non_gpu_review",
            detail="CPU-safe CI only; no GPU scorecard or queue entry required.",
            label=READY_NON_GPU_LABEL,
            kind=kind,
        )

    if not has_scorecard(pr.body):
        return _blocking_request_outcome(
            pr,
            comments,
            reason="scorecard",
            active_action="needs_scorecard",
            active_detail="PR body is missing a filled-in scorecard (see CONTRIBUTING.md)",
            label="status:needs-scorecard",
            kind=kind,
            close_detail=(
                "Closing this PR because the requested scorecard was not added within "
                "12 hours of the bot reminder. Open a fresh feat/strategy PR with a "
                "filled-in scorecard."
            ),
            now=now,
        )

    if run_eval is None:
        return GateOutcome(pr.number, "eval_pending",
                           detail="gate chain passed; GPU eval runner is not "
                                  "wired up yet in this repo (Phase 2)",
                           label=GPU_QUEUE_LABEL,
                           kind=kind)

    result = run_eval(pr)
    return GateOutcome(pr.number, "evaluated", detail=json.dumps(result), kind=kind)


def _apply(client: GitHubClient, pr: PRInfo, outcome: GateOutcome, comments: list) -> None:
    """Perform the write action implied by ``outcome``. Only called from
    :func:`run_once` when ``dry_run=False`` -- not the default in Phase 1."""
    if outcome.action == "close_blocked":
        client.close_pr(pr.number, outcome.detail)
    elif outcome.action == "close_excess_open_pr":
        client.close_pr(pr.number, outcome.detail)
    elif outcome.action == "needs_merge_conflict_resolution":
        if merge_conflict_comment_time(comments, pr.head_sha) is None:
            stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            client.post_comment(
                pr.number,
                f"<!-- cco-merge-conflict:{pr.head_sha}:{stamp} -->\n"
                "This PR currently has merge conflicts with the base branch. "
                "Please resolve them within 12 hours or the bot will close the PR automatically.",
            )
        client.remove_label(pr.number, GPU_QUEUE_LABEL)
        client.remove_label(pr.number, READY_NON_GPU_LABEL)
        client.remove_label(pr.number, NEEDS_PR_KIND_LABEL)
        client.remove_label(pr.number, "status:needs-scorecard")
    elif outcome.action == "close_stale_merge_conflict":
        client.close_pr(pr.number, outcome.detail)
    elif outcome.action == "close_coding_agent_coauthor":
        client.close_pr(pr.number, outcome.detail)
    elif outcome.action == "close_stale_blocking_request":
        client.close_pr(pr.number, outcome.detail)
    elif outcome.action == "copycat_block":
        client.add_label(pr.number, "copycat")
        client.post_comment(pr.number, f"Closed as a copycat submission: {outcome.detail}")
        client.close_pr(pr.number, "copycat")
    elif outcome.action == "copycat_warn":
        client.add_label(pr.number, "copycat-warn")
        client.post_comment(pr.number, f"Flagged for maintainer review: {outcome.detail}")
    elif outcome.action == "needs_pr_kind":
        client.add_label(pr.number, NEEDS_PR_KIND_LABEL)
        client.remove_label(pr.number, GPU_QUEUE_LABEL)
        client.remove_label(pr.number, "status:needs-scorecard")
        client.remove_label(pr.number, READY_NON_GPU_LABEL)
        if blocking_change_comment_time(comments, pr.head_sha, "pr-kind") is None:
            stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            client.post_comment(
                pr.number,
                f"<!-- cco-blocking-change:pr-kind:{pr.head_sha}:{stamp} -->\n"
                + NEEDS_PR_KIND_MARKER.format(sha=pr.head_sha)
                + "\nPlease mark this PR as either `fix`/`bug` or "
                  "`feat`/`strategy` in the title or PR template. "
                  "If this same head SHA is not updated within 12 hours, "
                  "the bot will close the PR automatically.",
            )
    elif outcome.action == "needs_scorecard":
        client.add_label(pr.number, "status:needs-scorecard")
        client.remove_label(pr.number, NEEDS_PR_KIND_LABEL)
        client.remove_label(pr.number, GPU_QUEUE_LABEL)
        client.remove_label(pr.number, READY_NON_GPU_LABEL)
        if blocking_change_comment_time(comments, pr.head_sha, "scorecard") is None:
            stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            client.post_comment(
                pr.number,
                f"<!-- cco-blocking-change:scorecard:{pr.head_sha}:{stamp} -->\n"
                + NEEDS_SCORECARD_MARKER.format(sha=pr.head_sha)
                + "\nPlease add a filled-in scorecard from "
                  "`python -m eval` (see CONTRIBUTING.md). "
                  "If this same head SHA is not updated within 12 hours, "
                  "the bot will close the PR automatically.",
            )
    elif outcome.action == "protected_path":
        client.add_label(pr.number, PROTECTED_PATH_LABEL)
        client.remove_label(pr.number, READY_NON_GPU_LABEL)
        client.remove_label(pr.number, NEEDS_PR_KIND_LABEL)
        client.remove_label(pr.number, "status:needs-scorecard")
        client.remove_label(pr.number, GPU_QUEUE_LABEL)
        if blocking_change_comment_time(comments, pr.head_sha, "protected-path") is None:
            stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            client.post_comment(
                pr.number,
                f"<!-- cco-blocking-change:protected-path:{pr.head_sha}:{stamp} -->\n"
                "This PR touches maintainer-owned files and will not enter the GPU "
                f"queue: {outcome.detail}. Split miner submissions so scoring "
                "changes stay outside protected paths. If this same head SHA is not "
                "updated within 12 hours, the bot will close the PR automatically.",
            )
    elif outcome.action == "non_gpu_review":
        client.add_label(pr.number, READY_NON_GPU_LABEL)
        client.remove_label(pr.number, NEEDS_PR_KIND_LABEL)
        client.remove_label(pr.number, "status:needs-scorecard")
        client.remove_label(pr.number, GPU_QUEUE_LABEL)
    elif outcome.action == "eval_pending":
        client.remove_label(pr.number, "status:needs-scorecard")
        client.remove_label(pr.number, NEEDS_PR_KIND_LABEL)
        client.remove_label(pr.number, READY_NON_GPU_LABEL)
        client.add_label(pr.number, GPU_QUEUE_LABEL)
        if not already_queued(comments, pr.head_sha):
            client.post_comment(
                pr.number,
                IDEMPOTENCY_MARKER.format(sha=pr.head_sha)
                + "\nGate chain passed. This PR is queued for the next batched "
                  "GPU evaluation window.",
            )
    elif outcome.action == "already_evaluated":
        client.remove_label(pr.number, "status:needs-scorecard")
        client.remove_label(pr.number, NEEDS_PR_KIND_LABEL)
        client.remove_label(pr.number, READY_NON_GPU_LABEL)
        client.remove_label(pr.number, GPU_QUEUE_LABEL)
    # skip_draft / evaluated: nothing to write here.
    # ("evaluated" posts its own scorecard comment from within run_eval /
    # eval.runner once Phase 2 wires that in -- not this function's job.)


def _queue_record(pr: PRInfo, outcome: GateOutcome, position: int | None = None) -> dict:
    record = {
        "pr": pr.number,
        "title": pr.title,
        "author": pr.author,
        "head_sha": pr.head_sha,
        "url": pr.url,
        "updated_at": pr.updated_at,
        "kind": outcome.kind,
        "gpu_required": outcome.kind == "feat",
        "state": outcome.action,
        "detail": outcome.detail,
    }
    if position is not None:
        record["position"] = position
    if outcome.label:
        record["label"] = outcome.label
    return record


def build_queue_dashboard(prs: list[PRInfo], outcomes: list[GateOutcome]) -> dict:
    """Build the live PR queue feed consumed by an external dashboard UI.

    This is deliberately separate from eval.ledger's sealed-results dashboard:
    queued PRs are not final evaluations, they are the worklist for the next
    sequential GPU batch.
    """
    by_pr = {o.pr: o for o in outcomes}
    open_prs = [_queue_record(pr, by_pr[pr.number]) for pr in prs if pr.number in by_pr]
    ready_prs = sorted(
        (
            pr for pr in prs
            if pr.number in by_pr and by_pr[pr.number].action in GPU_QUEUE_READY_ACTIONS
        ),
        key=lambda pr: (pr.updated_at or "", pr.number),
    )
    queue = [
        _queue_record(pr, by_pr[pr.number], position=i + 1)
        for i, pr in enumerate(ready_prs)
    ]
    return {
        "updated": "",
        "gpu_policy": {
            "mode": "batched-sequential",
            "cadence": "one or two maintainer-controlled GPU windows per day",
            "gpu_required_for": ["feat"],
            "ready_label": GPU_QUEUE_LABEL,
        },
        "queue": queue,
        "open_prs": open_prs,
    }


def _without_updated(data: dict) -> dict:
    clean = dict(data)
    clean.pop("updated", None)
    return clean


def write_queue_dashboard(path: str | Path, data: dict) -> bool:
    """Write dashboard queue data only when the queue state changed."""
    p = Path(path)
    old = None
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
    if old is not None and _without_updated(old) == _without_updated(data):
        return False

    out = dict(data)
    out["updated"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def run_once(
    client: GitHubClient,
    dry_run: bool = True,
    run_eval=None,
    dashboard_data: str | None = None,
) -> list:
    """Fetch state via ``client``, decide via :func:`process_pr` for every
    open PR (oldest first), then apply the outcome unless ``dry_run``."""
    blocked = load_blocked_contributors()
    all_prs = sorted(client.list_prs("all"), key=lambda p: p.number)
    open_prs = sorted(client.list_prs("open"), key=lambda p: (p.updated_at or "", p.number))
    excess_pr_numbers = excess_open_prs(open_prs)
    now = datetime.now(timezone.utc)

    diff_by_pr = {}
    fp_by_pr = {}
    commit_messages_by_pr = {}
    for p in all_prs:
        diff_by_pr[p.number] = client.get_diff(p.number)
        fp_by_pr[p.number] = copycat_guard.fingerprint(diff_by_pr[p.number])
        commit_messages_by_pr[p.number] = client.get_commit_messages(p.number)

    outcomes = []
    for pr in open_prs:
        diff = diff_by_pr.get(pr.number, client.get_diff(pr.number))
        comments = client.get_comments(pr.number)
        # Every earlier PR (any state -- open, closed, or merged) is a valid
        # copycat comparison target; PR number order is creation order.
        originals = [(p.author, fp_by_pr[p.number]) for p in all_prs if p.number < pr.number]
        outcome = process_pr(
            pr,
            diff,
            comments,
            blocked,
            originals,
            excess_pr_numbers,
            commit_messages_by_pr.get(pr.number, ""),
            now,
            run_eval,
        )
        outcomes.append(outcome)

        if not dry_run:
            _apply(client, pr, outcome, comments)

    if dashboard_data:
        write_queue_dashboard(dashboard_data, build_queue_dashboard(open_prs, outcomes))
    return outcomes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval.pr_bot",
        description="Gate open PRs (draft/blocked/copycat/scorecard checks). "
                    "GPU evaluation itself is stubbed until Phase 2.",
    )
    p.add_argument("--repo", default=REPO_DEFAULT)
    p.add_argument("--write", action="store_true",
                   help="actually write labels/comments/close actions back to "
                        "GitHub. Omit this in Phase 1; dry-run is the safe "
                        "default until Phase 3 wires up a live bot identity.")
    p.add_argument("--dashboard-data",
                   help="optional path to write the live PR GPU queue feed")
    args = p.parse_args(argv)

    client = GitHubClient(args.repo)
    outcomes = run_once(client, dry_run=not args.write,
                        dashboard_data=args.dashboard_data)
    for o in outcomes:
        line = f"PR #{o.pr}: {o.action}"
        if o.detail:
            line += f" -- {o.detail}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
