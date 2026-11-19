"""Behavioural tests for the shadow-atlas helpers.

Every test is red on the shipped stubs: each one calls the public API and the
only failure mode is ``NotImplementedError``.  Do not edit or weaken the tests.
"""

import pytest

from shadow.atlas import (
    AtlasError,
    align_rect,
    allocate_cascade,
    defragment,
    evict_batch,
    free,
    grow,
    pack,
    set_filtering,
)

ALIGN = 16


def _make_atlas(size=(64, 64), max_size=(64, 64), alignment=ALIGN):
    return {
        "size": tuple(size),
        "max_size": tuple(max_size),
        "alignment": alignment,
        "items": {},
        "holes": [],
        "filtering": {},
    }


def _overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _area(rect):
    return rect[2] * rect[3]


def _used_area(atlas):
    return sum(_area(rect) for rect in atlas["items"].values())


def test_pack_respects_alignment():
    requests = [("a", (17, 17)), ("b", (30, 20)), ("c", (40, 40))]
    result = pack(requests, (128, 128), 16)
    assert result["unplaced"] == []
    placements = result["placements"]
    assert set(placements) == {"a", "b", "c"}
    for x, y, w, h in placements.values():
        assert x % 16 == 0 and y % 16 == 0
        assert w % 16 == 0 and h % 16 == 0


def test_align_rect_rounds_up_to_boundary():
    assert align_rect(16, 16, 16) == (16, 16)
    assert align_rect(17, 33, 16) == (32, 48)
    assert align_rect(1, 1, 64) == (64, 64)
    for w in (1, 7, 17, 63):
        aw, ah = align_rect(w, w + 5, 32)
        assert aw >= w and ah >= w + 5
        assert aw % 32 == 0 and ah % 32 == 0


def test_allocate_cascade_splits_budget_by_coverage():
    result = allocate_cascade([("near", 0.5), ("mid", 0.3), ("far", 0.2)], 1024)
    allocations = result["allocations"]
    assert set(allocations) == {"near", "mid", "far"}
    assert allocations["near"] > allocations["mid"] > allocations["far"]
    assert allocations["near"] >= 512
    assert sum(allocations.values()) <= 1024
    assert result["used"] == sum(allocations.values())


def test_allocate_cascade_scales_depth_offset_with_resolution():
    result = allocate_cascade([("near", 0.75), ("far", 0.25)], 2048)
    allocations = result["allocations"]
    offsets = result["depth_offsets"]
    assert allocations["near"] > allocations["far"]
    assert offsets["near"] < offsets["far"]
    assert offsets["far"] / offsets["near"] == pytest.approx(
        allocations["near"] / allocations["far"]
    )
    assert offsets["near"] * allocations["near"] == pytest.approx(
        offsets["far"] * allocations["far"]
    )


def test_grow_stops_at_max_size():
    too_big = _make_atlas(size=(64, 64), max_size=(128, 128))
    blocked = grow(too_big, ("huge", (256, 256)))
    assert blocked["ok"] is False
    assert blocked["size"] == (64, 64)
    assert too_big["size"] == (64, 64)
    assert too_big["items"] == {}

    wide = _make_atlas(size=(64, 64), max_size=(128, 128))
    assert grow(wide, ("wide", (192, 64)))["ok"] is False
    assert wide["size"] == (64, 64)

    grows = _make_atlas(size=(64, 64), max_size=(256, 256))
    fitted = grow(grows, ("big", (100, 100)))
    assert fitted["ok"] is True
    assert fitted["size"] == (128, 128)
    assert fitted["rect"] == (0, 0, 112, 112)


def test_free_leaves_reusable_hole():
    atlas = _make_atlas(size=(64, 64), max_size=(64, 64))
    first = grow(atlas, ("a", (32, 32)))
    assert first["ok"] is True
    assert first["rect"] == (0, 0, 32, 32)
    assert grow(atlas, ("b", (32, 32)))["ok"] is True

    freed = free(atlas, "a")
    assert freed["ok"] is True
    assert freed["hole"] == (0, 0, 32, 32)
    assert "a" not in atlas["items"]

    reused = grow(atlas, ("c", (32, 32)))
    assert reused["ok"] is True
    assert reused["rect"] == (0, 0, 32, 32)
    assert atlas["size"] == (64, 64)


def test_defragment_preserves_used_area():
    atlas = _make_atlas(size=(128, 128), max_size=(128, 128))
    for name, size in (
        ("a", (32, 32)),
        ("b", (32, 32)),
        ("c", (32, 32)),
        ("d", (32, 48)),
    ):
        assert grow(atlas, (name, size))["ok"] is True
    free(atlas, "b")
    free(atlas, "d")

    used_before = _used_area(atlas)
    holes_before = len(atlas["holes"])
    assert holes_before >= 2

    result = defragment(atlas)
    assert result["ok"] is True
    assert result["used"] == used_before
    assert _used_area(atlas) == used_before
    assert len(atlas["holes"]) < holes_before

    rects = list(atlas["items"].values())
    for x, y, w, h in rects:
        assert 0 <= x and 0 <= y and x + w <= 128 and y + h <= 128
    for index, rect in enumerate(rects):
        for other in rects[index + 1 :]:
            assert not _overlaps(rect, other)


def test_set_filtering_is_per_rect():
    atlas = _make_atlas(size=(128, 128), max_size=(128, 128))
    assert grow(atlas, ("a", (16, 16)))["ok"] is True
    assert grow(atlas, ("b", (16, 16)))["ok"] is True

    first = set_filtering(atlas, "a", "nearest")
    assert first["mode"] == "nearest"
    second = set_filtering(atlas, "b", "linear")
    assert second["filtering"] == {"a": "nearest", "b": "linear"}


def test_set_filtering_rejects_out_of_bounds_rect():
    atlas = _make_atlas(size=(64, 64), max_size=(64, 64))
    assert grow(atlas, ("a", (32, 32)))["ok"] is True

    atlas["items"]["bad"] = (48, 48, 32, 32)
    with pytest.raises(AtlasError):
        set_filtering(atlas, "bad", "linear")

    atlas["items"]["neg"] = (-16, 0, 16, 16)
    with pytest.raises(AtlasError):
        set_filtering(atlas, "neg", "nearest")

    assert "bad" not in atlas["filtering"]
    assert "neg" not in atlas["filtering"]


def test_evict_batch_reports_partial_failure():
    atlas = _make_atlas(size=(128, 128), max_size=(128, 128))
    assert grow(atlas, ("a", (16, 16)))["ok"] is True
    assert grow(atlas, ("b", (16, 16)))["ok"] is True
    assert grow(atlas, ("c", (16, 16)))["ok"] is True
    a_rect = atlas["items"]["a"]

    result = evict_batch(atlas, ["a", "missing", "c"], 4096)
    assert result["ok"] is False
    assert result["evicted"] == ["a"]
    assert "a" not in atlas["items"]
    assert "b" in atlas["items"] and "c" in atlas["items"]
    assert any(hole == a_rect for hole in atlas["holes"])


def test_pack_is_deterministic():
    requests = [("a", (17, 17)), ("b", (32, 32)), ("c", (9, 40)), ("d", (48, 16))]
    first = pack(requests, (128, 128), 16)
    second = pack(requests, (128, 128), 16)
    assert first == second


def test_pack_empty_requests():
    assert pack([], (64, 64), 16) == {"placements": {}, "unplaced": []}


def test_align_rect_rejects_non_power_of_two_alignment():
    for bad in (0, 24, -16, 48):
        with pytest.raises(AtlasError):
            align_rect(10, 10, bad)


def test_pack_rects_do_not_overlap_and_stay_inside():
    requests = [("a", (17, 17)), ("b", (33, 9)), ("c", (48, 48)), ("d", (15, 31))]
    result = pack(requests, (128, 128), 16)
    assert result["unplaced"] == []
    rects = list(result["placements"].values())
    assert len(rects) == len(requests)
    for x, y, w, h in rects:
        assert 0 <= x and 0 <= y and x + w <= 128 and y + h <= 128
    for index, rect in enumerate(rects):
        for other in rects[index + 1 :]:
            assert not _overlaps(rect, other)


def test_grow_allows_exact_max_size_once():
    atlas = _make_atlas(size=(128, 128), max_size=(128, 128))
    first = grow(atlas, ("a", (128, 128)))
    assert first["ok"] is True
    assert first["rect"] == (0, 0, 128, 128)
    assert first["size"] == (128, 128)

    second = grow(atlas, ("b", (128, 128)))
    assert second["ok"] is False
    assert atlas["size"] == (128, 128)
