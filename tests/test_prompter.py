"""Unit tests for MiniMax-H3 prompt normalization and validation."""

from __future__ import annotations

from loki_remake.gpuniq.prompter import normalize_ascii_punctuation, validate_minimax_prompt

VALID_SWAP_PROMPT = """subject_definitions:
<Video 1> is the source video: one continuous take of a man tossing pizza dough.
<Picture 1> is a frame of the target video from [Shot 1].
<Subject 1> is Pete in <Picture 1>: a golden pineapple mascot.
He replaces the man in the grey hoodie in <Video 1>.

summary:
[video editing + character replacement] The target video is an edited version of <Video 1>
in which the man becomes <Subject 1>. Motion and timing of <Video 1> are kept.

retention_analysis:
<Video 1> (source video editing): fully_preserved - single shot, motion, and timing.
<Picture 1> (appears in [Shot 1]): reference - a frame of the target video.
<Subject 1> (appears in [Shot 1]): fully_preserved - mascot suit stays identical.

detailed_description:
A sunny handheld outdoor look with crisp daylight.
[Shot 1] From 00:00.000 to 00:09.375, <Subject 1> faces the camera and stretches pizza dough,
matching <Picture 1> at 00:01.000.

overall_soundscape:
The source soundtrack continues with park ambience.

non_diegetic_music:
N/A"""

VALID_EDIT_PROMPT = """subject_definitions:
<Video 1> is the source video for the target video edit.
<Picture 1> is a frame of the target video, exactly as the target video shows it at that moment.
<Subject 1> is the red leather jacket worn by the man in <Picture 1>.
<Subject 2> is the park setting and pizza dough from <Video 1>.

summary:
[video editing] The target video is an edited version of <Video 1> in which the man
wears <Subject 1> instead of a grey hoodie while tossing dough in <Subject 2>.

retention_analysis:
<Video 1> (source video editing): fully_preserved - camera motion, framing, and timing
are copied frame for frame.
<Picture 1> (target keyframe): fully_preserved - target passes through this frame.
<Subject 1> (appears in [Shot 1]): new - red leather jacket appears in every frame.
<Subject 2> (appears in [Shot 1]): fully_preserved - park background remains unchanged.

detailed_description:
A realistic sunny outdoor handheld look.
[Shot 1] From 00:00.000 to 00:09.375, the man wears <Subject 1> as he tosses dough
in <Subject 2>, matching <Picture 1> at 00:00.500.

overall_soundscape:
The soundtrack of <Video 1>, unchanged.

non_diegetic_music:
N/A"""


def test_valid_swap_and_edit_prompts() -> None:
    res_swap = validate_minimax_prompt(
        VALID_SWAP_PROMPT, mode="swap", num_frames=226, num_pics=1, num_shots=1
    )
    assert res_swap.valid, res_swap.errors

    res_edit = validate_minimax_prompt(
        VALID_EDIT_PROMPT, mode="edit", num_frames=226, num_pics=1, num_shots=1
    )
    assert res_edit.valid, res_edit.errors


def test_ascii_normalization() -> None:
    raw = "```\nsubject_definitions:\n“Hello” — it’s a test…\n```"
    cleaned = normalize_ascii_punctuation(raw)
    assert cleaned == "subject_definitions:\n\"Hello\" - it's a test..."


def test_missing_or_reordered_headers() -> None:
    reordered = VALID_EDIT_PROMPT.replace("summary:\n", "").replace(
        "detailed_description:", "summary:\n[video editing]\ndetailed_description:"
    )
    res = validate_minimax_prompt(
        reordered, mode="edit", num_frames=226, num_pics=1, num_shots=1
    )
    assert not res.valid
    assert any("out of order" in err for err in res.errors)


def test_timestamp_past_end_and_dim_word() -> None:
    bad = VALID_EDIT_PROMPT.replace("00:09.375", "00:12.500").replace(
        "sunny outdoor", "dimly lit outdoor"
    )
    res = validate_minimax_prompt(bad, mode="edit", num_frames=226, num_pics=1, num_shots=1)
    assert not res.valid
    assert any("exceeds clip end" in err for err in res.errors)
    assert any("dim" in err.lower() for err in res.errors)
