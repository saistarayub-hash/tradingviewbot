"""Video ingestion: YouTube link → transcript text.

Strategy:
  1. Fetch the watch page and read the ``captionTracks`` endpoints directly
     (no API key, no heavy dependencies — just httpx).
  2. Request the ``fmt=json3`` timed-text payload and join the segments.
  3. If the network is unavailable (e.g. this sandbox has no YouTube access)
     the brain falls back to a clearly-labelled demo transcript so the whole
     pipeline can still be tried end-to-end.

For videos without captions you can plug in whisper transcription
(``pip install faster-whisper``) — see README.
"""
from __future__ import annotations

import html
import re

import httpx

YT_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:[^&\s]+&)*v=|shorts/|embed/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{6,20})"
)
_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# A realistic strategy-video transcript used ONLY when YouTube cannot be
# reached. Lets you test the whole pipeline offline.
DEMO_TRANSCRIPT = """Alright guys, welcome back. Today I want to show you my favourite
crypto scalping strategy for the fifteen minute chart. It's built around two
exponential moving averages: the nine EMA and the twenty one EMA. When the nine
EMA crosses above the twenty one EMA we have our long entry signal. Before we
take that trade we also want the RSI to be below fifty, that keeps us out of
overbought chop. For the exit, we simply wait for the nine EMA to cross back
below the twenty one EMA. And you can flip the whole system for shorts: when
the nine EMA crosses below the twenty one, that's your short signal, and you
cover when it crosses back above. Risk management is super important here: I
put my stop loss two ATR below entry, and I'm always looking for at least a
two to one risk reward target. And remember, never risk more than one percent
of your account on a single trade. That's the whole system, hope it helps."""
DEMO_NOTE = "YouTube unreachable — used built-in demo transcript"


class TranscriptUnavailable(Exception):
    pass


def extract_video_url(text: str) -> str | None:
    """Return the first YouTube URL found in free-form chat text, if any."""
    m = YT_RE.search(text or "")
    return m.group(0) if m else None


def video_id_from_url(url: str) -> str | None:
    m = YT_RE.search(url or "")
    return m.group(1) if m else None


def youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _caption_base_urls(page: str) -> list[str]:
    i = page.find('"captionTracks"')
    if i == -1:
        return []
    chunk = page[i : i + 8000]
    return re.findall(r'"baseUrl"\s*:\s*"([^"]+)"', chunk)


def _pick_caption(urls: list[str]) -> str | None:
    if not urls:
        return None
    for prefer in ("lang=en", "en&", "auto"):
        for u in urls:
            if prefer in u:
                return u
    return urls[0]


async def fetch_transcript(video_id: str) -> tuple[str, str]:
    """Return ``(transcript_text, source_label)`` for a YouTube video id."""
    async with httpx.AsyncClient(timeout=30, headers=_UA, follow_redirects=True) as client:
        resp = await client.get(f"https://www.youtube.com/watch?v={video_id}")
        resp.raise_for_status()
        page = resp.text

        base = _pick_caption(_caption_base_urls(page))
        if not base:
            raise TranscriptUnavailable(
                "No captions found on this video. Try a video that has subtitles, "
                "or install yt-dlp + faster-whisper for full audio transcription."
            )

        timed = await client.get(
            re.sub(r"fmt=[^&]+", "fmt=json3", base),
            headers={"Referer": f"https://www.youtube.com/watch?v={video_id}"},
        )
        timed.raise_for_status()
        data = timed.json()

    parts: list[str] = []
    for event in data.get("events", []):
        for seg in event.get("segs", []):
            txt = seg.get("utf8")
            if txt and txt.strip():
                parts.append(txt)
    text = html.unescape(" ".join(parts)).strip()
    if not text:
        raise TranscriptUnavailable("Caption track was empty.")
    return text, "youtube captions"


async def video_meta(video_id: str) -> dict:
    """Grab the video title from the watch page."""
    try:
        async with httpx.AsyncClient(timeout=20, headers=_UA, follow_redirects=True) as client:
            resp = await client.get(f"https://www.youtube.com/watch?v={video_id}")
            resp.raise_for_status()
            m = re.search(r"<title>(.*?)</title>", resp.text)
            title = html.unescape(m.group(1)).replace(" - YouTube", "").strip() if m else ""
    except Exception:
        title = ""
    return {
        "id": video_id,
        "title": title or f"Video {video_id}",
        "url": youtube_url(video_id),
    }


async def ingest_video(url: str) -> tuple[dict, str]:
    """Fetch metadata + transcript for a YouTube URL.

    Returns ``(video_dict, note)``. Falls back to the demo transcript when
    YouTube is unreachable, with a clearly-labelled note.
    """
    video_id = video_id_from_url(url)
    if not video_id:
        raise TranscriptUnavailable("That doesn't look like a YouTube link I can read.")
    meta = await video_meta(video_id)
    try:
        transcript, label = await fetch_transcript(video_id)
        note = label
    except TranscriptUnavailable:
        raise
    except Exception as exc:  # network blocked etc.
        transcript, note = DEMO_TRANSCRIPT, DEMO_NOTE + f" ({type(exc).__name__})"
    meta["transcript"] = transcript
    meta["transcript_source"] = note
    return meta, note
