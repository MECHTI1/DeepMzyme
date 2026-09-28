#!/usr/bin/env python3
"""DeepMzyme documentation contract checker (docs consolidation PLAN_v2, section 12.5).

Usage: <python> tools/check_docs_contract.py
Exit status 1 when any STRICT check fails; WARN lines are informational.

Raising a cap, relaxing or promoting a check, or allowing a new top-level
document needs the user's approval. Fix the text, not the checker.
Standard library only; nothing is read at import time. Checks the working tree.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

KB = 1000
# Size targets. They stay warnings until Job C sets final caps and strictness.
BYTE_CAPS = {
    "AGENTS.md": 12 * KB,
    "Plan.md": 35 * KB,
    "EXPERIMENT_STATUS.md": 6 * KB,
    "docs/README.md": 5 * KB,
}
DEFAULT_READ_CAP = 25 * KB
CAMPAIGN_README_MAX_LINES = 60

# Top-level documents. A new one requires merging or archiving an existing one.
ALLOWED_ROOT_DOCS = frozenset({
    "AGENTS.md", "CLAUDE.md", "EXPERIMENT_STATUS.md", "GEMINI.md", "Plan.md", "README.md",
})
ALLOWED_DOCS_TOP_LEVEL = frozenset({
    "BACKUP_AND_DATA_INVENTORY.md", "COLAB_GPU_RUNBOOK.md", "DATASETS.md",
    "EC_TRAINING_PIPELINE_PLAYBOOK.md", "EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md",
    "FOLLOW_UP_TECHNICAL_ISSUES.md", "GCP_GPU_RUNBOOK.md", "GENERALIZED_METAL_5FOLD_CV_GUIDE.md",
    "GETTING_STARTED.md", "GPU_EXECUTION_CASCADE_PLAYBOOK.md", "GPU_RUNTIME_EFFICIENCY_PLAN.md",
    "METAL_NOTEBOOK_CONFIGURATION_GUIDE.md", "METAL_TRAINING_PIPELINE_PLAYBOOK.md", "MOVED.md",
    "PARAMETER_FINDINGS.md", "README.md", "REMOTE_HOMOLOGY_ADDENDUM.md",
    "REPRODUCIBILITY_REMEDIATION_PLAN.md", "STRUCTURE_STORE.md", "VERY_EXACT_PMM_SETS_PLAN.md",
    "ZENODO_PINMYMETAL_EXACT_ION_LEVEL_REPRODUCIBILITY.md",
})

# Immutable evidence is never edited, so its links are not checked here.
IMMUTABLE_PREFIXES = ("docs/notebook_outputs/raw/", "docs/notebook_outputs/summaries/")

STATUS_PATH = "EXPERIMENT_STATUS.md"
STATUS_TITLE = "# DeepMzyme Current Experiment Status"
STATUS_SECTIONS = (
    "Current objective and stage",
    "Anchor and evidence state",
    "Dataset and test readiness",
    "Blockers and immediate next action",
    "History and update rule",
)
STATUS_SECTION_LINES = {
    "Current objective and stage": ("Current campaign:", "Other open campaigns:", "Stage:"),
    "Anchor and evidence state": ("Best validation result:",),
}
STATUS_BLOCKER_LEAD = ("Authorized now:", "GPU/VM:")
STATUS_CAVEATS_HEADING = "### Known caveats and open mismatches"
STATUS_LINE = re.compile(r"^Status: (active|paused|closed|planned) \(\d{4}-\d{2}-\d{2}[^)]*\)\s*$")

PLAYBOOK_HEADING = "### Exact standalone notebook block"
# SHA-256 of each playbook's first python block, recorded before Job A (inventory/job_b_before.json).
PLAYBOOK_FIRST_BLOCK_SHA256 = {
    "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md": "c85419aa866930f8c75a81bea6098aae5b413bd8b444e1aaa5579df9ebbffcda",
    "docs/EC_TRAINING_PIPELINE_PLAYBOOK.md": "00e87b8f4e272433aeb67c656b07fc68aae8a4739200c2bf77afcba7dd5c99ec",
}

FOLLOW_UP_PATH = "docs/FOLLOW_UP_TECHNICAL_ISSUES.md"
CORE_DOCS = (
    "AGENTS.md", "Plan.md", "EXPERIMENT_STATUS.md", "README.md", "docs/README.md",
    "docs/DATASETS.md", "docs/PARAMETER_FINDINGS.md", FOLLOW_UP_PATH, "docs/notebook_outputs/README.md",
)
DUPLICATE_MIN_CHARS = 200

FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
LINK = re.compile(r"!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^)\s]+))(?:\s+\"[^\"]*\")?\s*\)")
INLINE_CODE = re.compile(r"(`+)(.+?)\1")
HTML_ANCHOR = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"")


class Findings:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []

    def add(self, level: str, where: str, rule: str, message: str, fix: str) -> None:
        self.items.append((level, where, f"[{rule}] {message} — fix: {fix}"))

    def count(self, level: str) -> int:
        return sum(1 for item in self.items if item[0] == level)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def list_markdown(root: Path) -> list[str]:
    try:
        output = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.md"],
            capture_output=True, check=True,
        ).stdout.decode("utf-8")
        names = [name for name in output.split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        names = [str(p.relative_to(root)) for p in root.rglob("*.md") if ".git" not in p.parts]
    return sorted({name for name in names if (root / name).is_file()})


def text_of(root: Path, name: str) -> str:
    return (root / name).read_text(encoding="utf-8")


def blank_fences(lines: list[str]) -> list[str]:
    """Replace fenced-code lines with empty strings, keeping line numbers."""
    out, fence = [], None
    for line in lines:
        match = FENCE.match(line)
        if fence:
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence) \
                    and line.strip() == match.group(1):
                fence = None
            out.append("")
        elif match:
            fence = match.group(1)
            out.append("")
        else:
            out.append(line)
    return out


def github_slug(heading: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"<[^>]+>", "", text).replace("`", "").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors_of(root: Path, name: str, cache: dict[str, set[str]]) -> set[str]:
    if name not in cache:
        raw = text_of(root, name)
        seen: dict[str, int] = {}
        found = set(HTML_ANCHOR.findall(raw))
        for line in blank_fences(raw.split("\n")):
            match = HEADING.match(line)
            if match:
                slug = github_slug(match.group(2))
                count = seen.get(slug, 0)
                seen[slug] = count + 1
                found.add(slug if count == 0 else f"{slug}-{count}")
        cache[name] = found
    return cache[name]


def check_links(root: Path, names: list[str], findings: Findings) -> int:
    cache: dict[str, set[str]] = {}
    checked = 0
    for name in names:
        if name.startswith(IMMUTABLE_PREFIXES):
            continue
        source = root / name
        for number, line in enumerate(blank_fences(text_of(root, name).split("\n")), 1):
            line = INLINE_CODE.sub(lambda m: " " * len(m.group(0)), line)
            for match in LINK.finditer(line):
                target = unquote(match.group(1) or match.group(2))
                if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):
                    continue
                checked += 1
                path_part, _, fragment = target.partition("#")
                if path_part.startswith("/"):
                    dest = (root / path_part.lstrip("/")).resolve()
                else:
                    dest = (source.parent / path_part).resolve() if path_part else source.resolve()
                where = f"{name}:{number}"
                try:
                    relative = dest.relative_to(root.resolve()).as_posix()
                except ValueError:
                    findings.add("STRICT", where, "links", f"link leaves the repository: {target}",
                                 "link to a file inside the repository")
                    continue
                if not dest.exists():
                    findings.add("STRICT", where, "links", f"missing link target: {target}",
                                 "point the link at an existing file or record the move in docs/MOVED.md")
                    continue
                if fragment and dest.is_file() and dest.suffix.lower() == ".md":
                    if fragment not in anchors_of(root, relative, cache):
                        findings.add("STRICT", where, "anchors", f"missing anchor: {target}",
                                     "use an existing heading slug or keep the old anchor with <a id=...>")
    return checked


def check_allowed_files(names: list[str], findings: Findings) -> None:
    for name in names:
        parts = name.split("/")
        if len(parts) == 1 and name not in ALLOWED_ROOT_DOCS:
            findings.add("STRICT", name, "allowed-files", "new root Markdown document",
                         "merge it into an owner document or archive another in the same change, with approval")
        if len(parts) == 2 and parts[0] == "docs" and parts[1] not in ALLOWED_DOCS_TOP_LEVEL:
            findings.add("STRICT", name, "allowed-files", "new top-level docs/ document",
                         "merge or archive an existing document in the same change and get approval to allow it")


def status_sections(lines: list[str]) -> dict[str, list[tuple[int, str]]]:
    sections: dict[str, list[tuple[int, str]]] = {}
    current = None
    for number, line in enumerate(lines, 1):
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append((number, line))
    return sections


def check_status(root: Path, findings: Findings) -> str | None:
    """Check the approved STATUS layout; return the current campaign README path."""
    if not (root / STATUS_PATH).is_file():
        findings.add("STRICT", STATUS_PATH, "status", "file is missing",
                     "restore the root EXPERIMENT_STATUS.md (code and notebooks rely on it)")
        return None
    lines = blank_fences(text_of(root, STATUS_PATH).split("\n"))
    if lines[0].strip() != STATUS_TITLE:
        findings.add("STRICT", f"{STATUS_PATH}:1", "status", "unexpected title", f"use '{STATUS_TITLE}'")
    head = [line for line in lines[:6] if line.strip()]
    if not any(STATUS_LINE.match(line) for line in head):
        findings.add("STRICT", STATUS_PATH, "status", "missing 'Status: <state> (<date> ...)' line near the top",
                     "add 'Status: active|paused|closed|planned (YYYY-MM-DD reason)'")
    if not any(line.startswith("Last execution evidence:") for line in head):
        findings.add("STRICT", STATUS_PATH, "status", "missing 'Last execution evidence:' line",
                     "add the dated evidence line under the Status line")
    order = [line[3:].strip() for line in lines if line.startswith("## ")]
    if tuple(order) != STATUS_SECTIONS:
        findings.add("STRICT", STATUS_PATH, "status", f"sections are {order}",
                     "use exactly: " + "; ".join(STATUS_SECTIONS))
    sections = status_sections(lines)
    for section, prefixes in STATUS_SECTION_LINES.items():
        body = [line for _, line in sections.get(section, [])]
        for prefix in prefixes:
            if not any(line.startswith(prefix) for line in body):
                findings.add("STRICT", STATUS_PATH, "status", f"'{section}' lacks the '{prefix}' line",
                             f"add '{prefix} ...' to that section")
    blocker_body = [line for _, line in sections.get("Blockers and immediate next action", []) if line.strip()]
    for index, prefix in enumerate(STATUS_BLOCKER_LEAD):
        if len(blocker_body) <= index or not blocker_body[index].startswith(prefix):
            findings.add("STRICT", STATUS_PATH, "status",
                         f"blockers section must start with '{STATUS_BLOCKER_LEAD[0]}' then '{STATUS_BLOCKER_LEAD[1]}'",
                         f"put the '{prefix} ...' line in position {index + 1}")
    anchor_body = [line for _, line in sections.get("Anchor and evidence state", [])]
    if STATUS_CAVEATS_HEADING not in [line.strip() for line in anchor_body]:
        findings.add("STRICT", STATUS_PATH, "status", "missing caveats slot",
                     f"add '{STATUS_CAVEATS_HEADING}' with bullets (or '- None.')")
    else:
        start = [line.strip() for line in anchor_body].index(STATUS_CAVEATS_HEADING)
        bullets: list[str] = []
        for line in anchor_body[start + 1:]:
            if line.startswith("- "):
                bullets.append(line)
            elif bullets and line.startswith("  ") and line.strip():
                bullets[-1] += " " + line.strip()
        if not bullets:
            findings.add("STRICT", STATUS_PATH, "status", "caveats slot has no bullet",
                         "add caveat bullets or '- None.'")
        for bullet in bullets:
            if "FOLLOW_UP_TECHNICAL_ISSUES.md" not in bullet and bullet.strip() != "- None.":
                findings.add("WARN", STATUS_PATH, "status", f"caveat without a FOLLOW_UP link: {bullet[:60]}",
                             "link the tracking TECH-### issue")
    campaign_line = next((line for line in lines if line.startswith("Current campaign:")), "")
    match = re.search(r"\]\((docs/campaigns/[^/)]+/README\.md)\)", campaign_line)
    if not match:
        findings.add("STRICT", STATUS_PATH, "status", "'Current campaign:' line lacks a docs/campaigns/<id>/README.md link",
                     "link the active campaign README on that line")
        return None
    readme = match.group(1)
    if not (root / readme).is_file():
        findings.add("STRICT", STATUS_PATH, "status", f"current campaign README missing: {readme}",
                     "fix the link or restore the campaign folder")
        return None
    return readme


def check_campaigns(root: Path, names: list[str], findings: Findings) -> None:
    folders = {}
    for name in names:
        parts = name.split("/")
        if len(parts) == 4 and parts[:2] == ["docs", "campaigns"]:
            folders.setdefault("/".join(parts[:3]), set()).add(parts[3])
        if len(parts) == 5 and parts[:3] == ["docs", "archive", "campaigns"]:
            folders.setdefault("/".join(parts[:4]), set()).add(parts[4])
    for folder, files in sorted(folders.items()):
        archived = folder.startswith("docs/archive/")
        readme = f"{folder}/README.md"
        if "README.md" not in files:
            findings.add("STRICT", folder, "campaigns", "campaign folder without README.md",
                         "add a README with a 'Status:' line")
            continue
        lines = text_of(root, readme).split("\n")
        status = next((line for line in lines[:8] if line.startswith("Status:")), "")
        match = STATUS_LINE.match(status)
        if not match:
            findings.add("STRICT", readme, "campaigns", "missing or malformed 'Status:' line near the top",
                         "use 'Status: active|paused|closed|planned (YYYY-MM-DD ...)'")
        elif archived and match.group(1) != "closed":
            findings.add("STRICT", readme, "campaigns", f"archived campaign has status '{match.group(1)}'",
                         "set 'Status: closed (...)' or move the folder back to docs/campaigns/")
        elif not archived and match.group(1) == "closed":
            findings.add("STRICT", readme, "campaigns", "closed campaign outside docs/archive/campaigns/",
                         "git mv the folder to docs/archive/campaigns/<id> and update STATUS")
        if len(lines) > CAMPAIGN_README_MAX_LINES + 1:
            findings.add("WARN", readme, "caps", f"{len(lines) - 1} lines (target {CAMPAIGN_README_MAX_LINES})",
                         "move detail into log.md or a linked owner")


def check_follow_up(root: Path, findings: Findings) -> None:
    if not (root / FOLLOW_UP_PATH).is_file():
        return
    lines = text_of(root, FOLLOW_UP_PATH).split("\n")
    seen: dict[str, int] = {}
    for number, line in enumerate(lines, 1):
        match = re.match(r"^## (TECH-\d{3})\b", line)
        if not match:
            continue
        issue = match.group(1)
        if issue in seen:
            findings.add("STRICT", f"{FOLLOW_UP_PATH}:{number}", "follow-up",
                         f"{issue} heading repeats line {seen[issue]}", "never reuse a TECH number")
        seen[issue] = number
        body = []
        for following in lines[number:]:
            if following.startswith("## "):
                break
            if following.strip():
                body.append(following)
        status = next((text for text in body[:3] if text.startswith("**Status:**")), None)
        stub = len(body) <= 3 and any("issues_resolved.md" in text for text in body)
        if status is None and not stub:
            findings.add("STRICT", f"{FOLLOW_UP_PATH}:{number}", "follow-up", f"{issue} has no '**Status:**' line",
                         "add '**Status:** ...' under the heading, or a one-line stub to the archive")
        elif status and re.match(r"\*\*Status:\*\*\s*(Resolved|Fixed)\b", status) and not stub:
            findings.add("WARN", f"{FOLLOW_UP_PATH}:{number}", "follow-up", f"{issue} is resolved but kept in full",
                         "move it to docs/archive/issues_resolved.md with an anchor stub (Job C)")


def check_playbooks(root: Path, findings: Findings) -> None:
    for name, expected in PLAYBOOK_FIRST_BLOCK_SHA256.items():
        path = root / name
        if not path.is_file():
            findings.add("STRICT", name, "playbook", "playbook missing", "restore it at its fixed path (code reads it)")
            continue
        raw = path.read_bytes()
        if b"\r" in raw:
            findings.add("STRICT", name, "playbook", "CR characters found", "convert the file to LF line endings")
        text = raw.decode("utf-8")
        lines = text.split("\n")
        headings = [i for i, line in enumerate(lines) if line.strip() == PLAYBOOK_HEADING]
        fences = [i for i, line in enumerate(lines) if line.startswith("```python")]
        if len(headings) != 1:
            findings.add("STRICT", name, "playbook", f"'{PLAYBOOK_HEADING}' appears {len(headings)} times",
                         "keep exactly one such heading")
        elif not fences or headings[0] > fences[0]:
            findings.add("STRICT", name, "playbook", "a python fence precedes the standalone heading",
                         "do not add python fences above the standalone notebook block")
        blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
        if not blocks or hashlib.sha256(blocks[0].encode()).hexdigest() != expected:
            findings.add("STRICT", name, "playbook", "first python block changed",
                         "restore the block byte-for-byte; tests and the notebook execute it")


def check_caps(root: Path, findings: Findings) -> None:
    for name, cap in BYTE_CAPS.items():
        path = root / name
        if path.is_file() and path.stat().st_size > cap:
            findings.add("WARN", name, "caps", f"{path.stat().st_size:,} bytes (target {cap:,})",
                         "shorten or link to the owner; raising the target needs approval")


def paragraphs(text: str) -> list[str]:
    blocks, current = [], []
    for line in blank_fences(text.split("\n")) + [""]:
        if line.strip():
            current.append(line.strip())
        elif current:
            blocks.append(" ".join(current))
            current = []
    return [" ".join(block.split()) for block in blocks
            if not block.startswith(("|", "#")) and len(" ".join(block.split())) >= DUPLICATE_MIN_CHARS]


def check_duplicates(root: Path, names: list[str], findings: Findings) -> None:
    core = [name for name in CORE_DOCS if (root / name).is_file()]
    core += [name for name in names if re.match(r"^docs/campaigns/[^/]+/README\.md$", name)]
    owners: dict[str, str] = {}
    for name in core:
        for block in paragraphs(text_of(root, name)):
            first = owners.setdefault(block, name)
            if first != name:
                findings.add("WARN", name, "duplicates", f"paragraph also in {first}: {block[:60]}...",
                             "keep it in the owner document and link to it")


def default_read(root: Path, campaign_readme: str | None, findings: Findings) -> str:
    parts = ["AGENTS.md", STATUS_PATH] + ([campaign_readme] if campaign_readme else [])
    sizes = [(name, (root / name).stat().st_size) for name in parts if (root / name).is_file()]
    total = sum(size for _, size in sizes)
    if total > DEFAULT_READ_CAP:
        findings.add("WARN", "default read path", "caps", f"{total:,} bytes (target {DEFAULT_READ_CAP:,})",
                     "shorten AGENTS, STATUS or the campaign README; raising the target needs approval")
    listed = " + ".join(f"{name} ({size:,})" for name, size in sizes)
    return f"Default read path: {listed} = {total:,} bytes (target {DEFAULT_READ_CAP:,})"


def main(root: Path | None = None) -> int:
    root = (root or repo_root()).resolve()
    findings = Findings()
    names = list_markdown(root)
    checked = check_links(root, names, findings)
    check_allowed_files(names, findings)
    campaign_readme = check_status(root, findings)
    check_campaigns(root, names, findings)
    check_follow_up(root, findings)
    check_playbooks(root, findings)
    check_caps(root, findings)
    check_duplicates(root, names, findings)
    read_path = default_read(root, campaign_readme, findings)
    for level, where, message in findings.items:
        print(f"{level} {where}: {message}")
    print(read_path)
    strict, warn = findings.count("STRICT"), findings.count("WARN")
    print(f"docs contract: {len(names)} Markdown files, {checked} relative links checked; "
          f"{strict} strict failure(s), {warn} warning(s)")
    return 1 if strict else 0


if __name__ == "__main__":
    sys.exit(main())
