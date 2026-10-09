"""Load agent skills from a directory of SKILL.md files (YAML frontmatter + markdown body)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import SKILLS_DIR

_FRONTMATTER_END = "\n---\n"


@dataclass(frozen=True)
class SkillEntry:
    name: str
    description: str
    body: str
    path: Path


def _parse_skill_file(path: Path) -> SkillEntry:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing YAML frontmatter")

    end = text.find(_FRONTMATTER_END, 3)
    if end == -1:
        raise ValueError(f"{path}: unclosed frontmatter")

    raw_meta = yaml.safe_load(text[3:end]) or {}
    if not isinstance(raw_meta, dict):
        raise ValueError(f"{path}: frontmatter must be a mapping")

    name = str(raw_meta.get("name") or path.parent.name).strip()
    description = str(raw_meta.get("description") or "").strip()
    body = text[end + len(_FRONTMATTER_END) :].lstrip("\n")
    return SkillEntry(name=name, description=description, body=body, path=path)


class SkillLoader:
    """Discover skills under ``skill_dir`` and expose catalog / on-demand load."""

    def __init__(self, skill_dir: Path | str) -> None:
        self.skill_dir = Path(skill_dir)
        self._by_name: dict[str, SkillEntry] = {}

    def refresh(self) -> None:
        """Rescan ``skill_dir`` for ``**/SKILL.md`` files."""
        self._by_name.clear()
        if not self.skill_dir.is_dir():
            return

        for path in sorted(self.skill_dir.rglob("SKILL.md")):
            try:
                entry = _parse_skill_file(path)
            except (OSError, ValueError) as exc:
                print(f"[SkillLoader] skip {path}: {exc}")
                continue
            self._by_name[entry.name] = entry

    def _ensure_index(self) -> None:
        if not self._by_name:
            self.refresh()

    def catalog(self) -> str:
        """Short list of skill names and descriptions for the system prompt."""
        self._ensure_index()
        if not self._by_name:
            return "No skills are available."

        lines = [
            "Available skills (call load_skills with names when you need full instructions):",
        ]
        for name in sorted(self._by_name):
            desc = self._by_name[name].description or "(no description)"
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def load(self, names: list[str]) -> str:
        """Return full skill markdown for the requested names."""
        self._ensure_index()
        if not names:
            return "Error: load_skills requires at least one skill name."

        parts: list[str] = []
        missing: list[str] = []
        for name in names:
            entry = self._by_name.get(name)
            if entry is None:
                missing.append(name)
                continue
            parts.append(f"# Skill: {entry.name}\n\n{entry.body.rstrip()}\n")

        out: list[str] = []
        if missing:
            known = ", ".join(sorted(self._by_name)) or "(none)"
            out.append(
                f"Unknown skill(s): {', '.join(missing)}. Known skills: {known}."
            )
        if parts:
            out.append("\n---\n\n".join(parts))
        return "\n".join(out).strip() or "No skills loaded."


_default_loader: SkillLoader | None = None


def get_skill_loader(skill_dir: Path | str | None = None) -> SkillLoader:
    """Process-wide default loader (lazy singleton) for tools and system prompt."""
    global _default_loader
    if skill_dir is not None:
        return SkillLoader(skill_dir)
    if _default_loader is None:
        _default_loader = SkillLoader(SKILLS_DIR)
    return _default_loader


def build_system_prompt(base: str) -> str:
    return f"{base}\n\n{get_skill_loader().catalog()}"
