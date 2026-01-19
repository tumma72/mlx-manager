#!/usr/bin/env python3
"""Update version in pyproject.toml using tomlkit to preserve formatting."""

import sys
from pathlib import Path

try:
    import tomlkit
except ImportError:
    # Fallback to regex if tomlkit not available
    import re

    def update_version_regex(pyproject_path: Path, version: str) -> bool:
        """Fallback: update version using regex (risky but works)."""
        content = pyproject_path.read_text()

        # Match version = "x.y.z" in [project] section
        pattern = r'^(version\s*=\s*")[^"]*(")'
        new_content, count = re.subn(pattern, rf'\g<1>{version}\g<2>', content, flags=re.MULTILINE)

        if count == 0:
            print(f"Warning: Could not find version in {pyproject_path}", file=sys.stderr)
            return False

        pyproject_path.write_text(new_content)
        return True

    if __name__ == "__main__":
        if len(sys.argv) != 3:
            print(f"Usage: {sys.argv[0]} <pyproject.toml> <version>", file=sys.stderr)
            sys.exit(1)

        pyproject_path = Path(sys.argv[1])
        version = sys.argv[2]

        if not pyproject_path.exists():
            print(f"Error: {pyproject_path} not found", file=sys.stderr)
            sys.exit(1)

        if update_version_regex(pyproject_path, version):
            print(f"✓ Updated {pyproject_path} to version {version} (regex fallback)")
        else:
            sys.exit(1)

    sys.exit(0)


def update_version_tomlkit(pyproject_path: Path, version: str) -> bool:
    """Update version using tomlkit (preserves formatting and comments)."""
    content = pyproject_path.read_text()
    doc = tomlkit.parse(content)

    if "project" not in doc:
        print(f"Error: No [project] section in {pyproject_path}", file=sys.stderr)
        return False

    doc["project"]["version"] = version
    pyproject_path.write_text(tomlkit.dumps(doc))
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <pyproject.toml> <version>", file=sys.stderr)
        sys.exit(1)

    pyproject_path = Path(sys.argv[1])
    version = sys.argv[2]

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found", file=sys.stderr)
        sys.exit(1)

    if update_version_tomlkit(pyproject_path, version):
        print(f"✓ Updated {pyproject_path} to version {version}")
    else:
        sys.exit(1)
