from __future__ import annotations

from pathlib import Path
import music21 as m21


def find_musescore_executable() -> Path | None:
    """
    Tries common MuseScore 4 locations on macOS.
    Returns the first existing executable path.
    """
    candidates = [
        # Common MuseScore 4 bundle names

        Path("C:/Program Files/MuseScore 4"),
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def configure_music21_musescore() -> Path:
    """
    Configures music21 to use MuseScore for direct PNG rendering.
    Raises RuntimeError if not found.
    """
    exe = find_musescore_executable()
    if exe is None:
        raise RuntimeError(
            "MuseScore executable not found. "
            "Open Finder → Applications and check the exact app name. "
            "Then check inside: MuseScore.app/Contents/MacOS/ ..."
        )

    # This will also validate the path exists (music21 does that)
    m21.environment.set("musescoreDirectPNGPath", str(exe))
    return exe


if __name__ == "__main__":
    exe = configure_music21_musescore()
    print("✅ music21 MuseScore path set to:", exe)
    print("music21 sees:", m21.environment.get("musescoreDirectPNGPath"))