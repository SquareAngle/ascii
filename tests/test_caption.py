"""Checks for the VLM captioning helpers (no model needed).

    python -m tests.test_caption
"""
import hashlib

from artscii.caption.vlm import build_messages, parse_response, piece_key, render_png

ART = " /\\_/\\\n( o.o )\n > ^ <"


def test_piece_key_matches_clean_dedup_key():
    lines = ART.split("\n")  # clean.dedup_key, inlined to avoid importing the scrapers
    assert piece_key(ART) == hashlib.sha1("".join(lines).replace(" ", "").encode()).hexdigest()


def test_render_grid_and_min_side():
    img = render_png(ART)
    assert max(img.size) >= 384
    wide = render_png("-" * 64 + "\n|" + " " * 62 + "|")
    assert wide.size[0] > wide.size[1] * 4  # aspect follows the character grid


def test_parse_response():
    good = '```json\n{"short": "a cat", "medium": "a cat face", "long": "a  cat\\nface."}\n```'
    assert parse_response(good) == {"short": "a cat", "medium": "a cat face", "long": "a cat face."}
    assert parse_response('{"unclear": true}') == {"unclear": True}
    conf = '{"short": "a", "medium": "b", "long": "c", "confidence": "Low"}'
    assert parse_response(conf)["confidence"] == "low"
    assert "confidence" not in parse_response(conf.replace("Low", "maybe"))
    assert parse_response('{"short": "a cat", "medium": ""}') is None
    assert parse_response("I think this is a cat") is None
    assert parse_response("[1, 2]") is None


def test_messages_carry_hints():
    msgs = build_messages({"category": "animals/cats", "title": "Tabby"}, "AAAA")
    text = msgs[0]["content"][1]["text"]
    assert 'category "animals/cats"' in text and 'title "Tabby"' in text
    assert "{" in text and "unclear" in text  # literal JSON braces survived .format
    usenet = build_messages({"source": "asciiart_eu_usenet", "category": "usenet > alt.ascii-art",
                             "title": "Re: newbie question"}, "AAAA")[0]["content"][1]["text"]
    assert "Re: newbie question" in usenet and "alt.ascii-art" not in usenet
    redundant = build_messages({"category": "animals/cats", "title": "cats"}, "AAAA")
    assert "title" not in redundant[0]["content"][1]["text"].split("Rules:")[0].split("Hints")[1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
