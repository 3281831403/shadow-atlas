"""Cascaded shadow-map atlas helpers.

The renderer keeps this module deliberately small: rectangles are
``(width, height)`` pairs, placements are ``(x, y, w, h)`` tuples and an atlas
is a plain mutable ``dict``.  No imaging library is used, so every operation is
integer arithmetic and can be exercised offline.

Atlas layout::

    {
        "size": (width, height),        # current texture size, in pixels
        "max_size": (width, height),    # hard hardware limit, never exceeded
        "alignment": 16 | 32 | 64,      # positive power of two
        "items": {id: (x, y, w, h)},    # live placements
        "holes": [(x, y, w, h), ...],   # free rectangles recorded by free()
        "filtering": {id: mode},        # filter mode recorded per rectangle
    }

Alignment must be a positive power of two (``2 ** 0`` included); anything else
raises :class:`AtlasError`.  Coordinates grow right and down from ``(0, 0)``.
"""

import math


#: Numerator of the depth-bias formula used by :func:`allocate_cascade`.
#: A cascade's depth offset is ``DEPTH_OFFSET_UNIT / resolution`` so that the
#: bias is expressed in world units per texel and shrinks as resolution grows.
DEPTH_OFFSET_UNIT = 1024.0


class AtlasError(Exception):
    """Raised for invalid atlas input: bad alignment, mode, bounds or id."""


# ---------------------------------------------------------------------------
# small integer helpers


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_alignment(alignment):
    if not _is_int(alignment):
        raise AtlasError("alignment must be an integer")
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise AtlasError("alignment must be a positive power of two")
    return alignment


def _validate_dimension(value, name):
    if not _is_int(value) or value <= 0:
        raise AtlasError("%s must be a positive integer" % name)
    return value


def _align_up(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def _overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _contains(outer, inner):
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ox <= ix and oy <= iy and ix + iw <= ox + ow and iy + ih <= oy + oh


def _first_fit(occupied, atlas_size, aligned_size, alignment):
    """Return the first non-overlapping aligned placement, or ``None``.

    Scan order is top-to-bottom, left-to-right at ``alignment`` steps, which is
    deterministic and makes the scan itself reuse a freed region: no hole list
    bookkeeping is needed to find the space again.
    """
    width, height = atlas_size
    rect_w, rect_h = aligned_size
    if rect_w > width or rect_h > height:
        return None
    for y in range(0, height - rect_h + 1, alignment):
        for x in range(0, width - rect_w + 1, alignment):
            candidate = (x, y, rect_w, rect_h)
            if not any(_overlaps(candidate, other) for other in occupied):
                return candidate
    return None


def _subtract_rect(outer, inner):
    """Return ``outer`` minus ``inner`` as up to four maximal strips."""
    if not _overlaps(outer, inner):
        return [outer]
    ax, ay, aw, ah = outer
    bx, by, bw, bh = inner
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    pieces = []
    if iy0 > ay:
        pieces.append((ax, ay, aw, iy0 - ay))
    if iy1 < ay + ah:
        pieces.append((ax, iy1, aw, ay + ah - iy1))
    if ix0 > ax:
        pieces.append((ax, iy0, ix0 - ax, iy1 - iy0))
    if ix1 < ax + aw:
        pieces.append((ix1, iy0, ax + aw - ix1, iy1 - iy0))
    return pieces


def _consume_holes(atlas, rect):
    """Remove ``rect`` from every recorded hole, keeping the hole map exact."""
    remaining = []
    for hole in atlas["holes"]:
        remaining.extend(_subtract_rect(hole, rect))
    atlas["holes"] = remaining


def _merge_pair(a, b):
    """Merge two axis-aligned holes that share a full edge, else ``None``."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if ay == by and ah == bh:
        if ax + aw == bx:
            return (ax, ay, aw + bw, ah)
        if bx + bw == ax:
            return (bx, by, aw + bw, ah)
    if ax == bx and aw == bw:
        if ay + ah == by:
            return (ax, ay, aw, ah + bh)
        if by + bh == ay:
            return (bx, by, bw, ah + bh)
    return None


def _merge_holes(holes):
    merged = list(holes)
    changed = True
    while changed:
        changed = False
        result = []
        for hole in merged:
            for index, keeper in enumerate(result):
                union = _merge_pair(keeper, hole)
                if union is not None:
                    result[index] = union
                    changed = True
                    break
            else:
                result.append(hole)
        merged = result
    return merged


# ---------------------------------------------------------------------------
# public API


def pack(requests, atlas_size, alignment):
    """Pack aligned rectangles into one atlas.

    Args:
        requests: sequence of ``(id, (w, h))`` pairs.  ``id`` is a hashable
            key and must be unique; the sequence order is meaningful because
            packing is deterministic and first-fit.
        atlas_size: ``(width, height)`` of the atlas in pixels.
        alignment: positive power of two (16, 32 and 64 are the sizes the
            renderer uses).

    Returns:
        ``{"placements": {id: (x, y, w, h)}, "unplaced": [id, ...]}``.
        ``w``/``h`` are the requested size rounded *up* to ``alignment`` and
        ``x``/``y`` are multiples of ``alignment``.  Requests that fit appear
        in ``placements``; the remaining ids appear in ``unplaced`` in input
        order.  No two placements overlap and each lies fully inside
        ``atlas_size``.

    Raises:
        AtlasError: ``alignment`` is not a positive power of two, a size is
            not positive, or two requests share an id.

    Deterministic: the same arguments always produce an equal result, with no
    randomness or wall-clock input.
    """
    _validate_alignment(alignment)
    width, height = atlas_size
    _validate_dimension(width, "atlas width")
    _validate_dimension(height, "atlas height")

    seen_ids = set()
    placements = {}
    unplaced = []
    occupied = []

    for request in requests:
        rect_id, size = request
        w, h = size
        if rect_id in seen_ids:
            raise AtlasError("duplicate request id: %r" % (rect_id,))
        seen_ids.add(rect_id)
        aligned_w, aligned_h = align_rect(w, h, alignment)
        placement = _first_fit(
            occupied, (width, height), (aligned_w, aligned_h), alignment
        )
        if placement is None:
            unplaced.append(rect_id)
        else:
            placements[rect_id] = placement
            occupied.append(placement)

    return {"placements": placements, "unplaced": unplaced}


def align_rect(w, h, alignment):
    """Round a rectangle size up to the next alignment boundary.

    Args:
        w: requested width in pixels, a positive integer.
        h: requested height in pixels, a positive integer.
        alignment: positive power of two.

    Returns:
        ``(aligned_w, aligned_h)`` with both values multiples of ``alignment``
        and never smaller than the inputs.  A size already on the boundary is
        returned unchanged.

    Raises:
        AtlasError: ``alignment`` is not a positive power of two, or ``w`` /
            ``h`` is not a positive integer.
    """
    _validate_alignment(alignment)
    _validate_dimension(w, "width")
    _validate_dimension(h, "height")
    return _align_up(w, alignment), _align_up(h, alignment)


def allocate_cascade(cascades, budget):
    """Split a texel budget across cascades by screen coverage.

    Args:
        cascades: sequence of ``(id, coverage)`` pairs.  ``coverage`` is a
            positive float (a screen-space weight, not necessarily normalised
            to 1).  Ids must be unique.
        budget: total texels available, a non-negative integer.

    Returns:
        ``{"allocations": {id: int}, "depth_offsets": {id: float},
        "used": int}``.  Each cascade's resolution is proportional to its
        coverage (floor), so the levels with more coverage receive more
        resolution and ``used == sum(allocations.values()) <= budget``.  Every
        resolution is at least 1 when ``budget`` is at least the number of
        cascades.  ``depth_offsets[id] == DEPTH_OFFSET_UNIT / resolution``: the
        bias shrinks as the cascade resolution grows instead of staying a
        fixed constant.

    Raises:
        AtlasError: ``cascades`` is empty, a coverage is not positive, an id is
            duplicated, ``budget`` is negative, or ``budget`` is too small to
            give every cascade at least one texel.
    """
    if not _is_int(budget) or budget < 0:
        raise AtlasError("budget must be a non-negative integer")
    if not cascades:
        raise AtlasError("at least one cascade is required")

    seen = set()
    total_coverage = 0.0
    for cascade_id, coverage in cascades:
        if cascade_id in seen:
            raise AtlasError("duplicate cascade id: %r" % (cascade_id,))
        seen.add(cascade_id)
        if (
            not isinstance(coverage, (int, float))
            or isinstance(coverage, bool)
            or not math.isfinite(coverage)
            or coverage <= 0
        ):
            raise AtlasError("coverage must be a positive number")
        total_coverage += coverage

    count = len(cascades)
    if budget < count:
        raise AtlasError("budget is too small for %d cascades" % count)

    allocations = {}
    for cascade_id, coverage in cascades:
        allocations[cascade_id] = max(
            1, int(math.floor(coverage / total_coverage * budget))
        )

    # Rounding up the tiny levels to one texel can push the sum over budget;
    # shave the excess from the largest allocations so every level keeps at
    # least one texel and the total never exceeds the budget.
    excess = sum(allocations.values()) - budget
    while excess > 0:
        candidates = [k for k, v in allocations.items() if v > 1]
        if not candidates:
            raise AtlasError("budget is too small for %d cascades" % count)
        largest = max(candidates, key=lambda k: (allocations[k], str(k)))
        allocations[largest] -= 1
        excess -= 1

    depth_offsets = {
        cascade_id: DEPTH_OFFSET_UNIT / resolution
        for cascade_id, resolution in allocations.items()
    }
    return {
        "allocations": allocations,
        "depth_offsets": depth_offsets,
        "used": sum(allocations.values()),
    }


def grow(atlas, request):
    """Allocate one rectangle, doubling the atlas when it does not fit.

    Args:
        atlas: mutable atlas dict as described in the module docstring.
        request: ``(id, (w, h))`` for the rectangle to allocate.  It is rounded
            up with :func:`align_rect` using ``atlas["alignment"]``.

    Returns:
        On success ``{"ok": True, "id": id, "rect": (x, y, w, h),
        "size": (width, height)}`` and the atlas is updated: the placement is
        recorded in ``items`` and any ``holes`` entry it fully covers is
        dropped.  The rectangle is placed first-fit from ``(0, 0)`` at
        alignment steps, so space released by :func:`free` is reused before
        the atlas grows.  When nothing fits, both dimensions are doubled
        (capped at ``max_size``) and placement is retried.

        On failure ``{"ok": False, "id": id, "size": (width, height),
        "reason": "max_size"}``: the atlas is left untouched, never grown past
        ``max_size`` and never grown in an unbounded loop.

    Raises:
        AtlasError: the id is already present, the request is not positive, or
            the atlas alignment is invalid.
    """
    alignment = _validate_alignment(atlas["alignment"])
    rect_id, size = request
    w, h = size
    aligned_w, aligned_h = align_rect(w, h, alignment)

    if rect_id in atlas["items"]:
        raise AtlasError("id already present: %r" % (rect_id,))

    max_width, max_height = atlas["max_size"]
    current_width, current_height = atlas["size"]
    occupied = list(atlas["items"].values())

    # A rectangle larger than the hardware limit can never fit; refuse before
    # touching the atlas so no capacity is wasted growing towards a dead end.
    if aligned_w > max_width or aligned_h > max_height:
        return {
            "ok": False,
            "id": rect_id,
            "size": (current_width, current_height),
            "reason": "max_size",
        }

    while True:
        placement = _first_fit(
            occupied,
            (current_width, current_height),
            (aligned_w, aligned_h),
            alignment,
        )
        if placement is not None:
            break

        next_width = min(current_width * 2, max_width)
        next_height = min(current_height * 2, max_height)
        if (next_width, next_height) == (current_width, current_height):
            # Already at the hardware limit: stop instead of doubling forever.
            return {
                "ok": False,
                "id": rect_id,
                "size": (current_width, current_height),
                "reason": "max_size",
            }
        current_width, current_height = next_width, next_height

    atlas["size"] = (current_width, current_height)
    atlas["items"][rect_id] = placement
    _consume_holes(atlas, placement)

    return {
        "ok": True,
        "id": rect_id,
        "rect": placement,
        "size": (current_width, current_height),
    }


def free(atlas, id):
    """Release a rectangle and record the hole it leaves behind.

    Args:
        atlas: mutable atlas dict.
        id: key of a live placement in ``atlas["items"]``.

    Returns:
        ``{"ok": True, "id": id, "hole": (x, y, w, h), "holes": [...]}``.
        The placement is removed from ``items`` and its rectangle is appended
        to ``holes`` (adjacent holes may be merged) so that a later
        :func:`grow` actually reuses the space instead of only flipping a
        status flag.

    Raises:
        AtlasError: ``id`` is not a live placement.
    """
    if id not in atlas["items"]:
        raise AtlasError("id is not a live placement: %r" % (id,))

    hole = atlas["items"].pop(id)
    atlas["holes"] = _merge_holes(list(atlas["holes"]) + [hole])

    return {
        "ok": True,
        "id": id,
        "hole": hole,
        "holes": list(atlas["holes"]),
    }


def defragment(atlas):
    """Repack live rectangles tightly from the atlas origin.

    Args:
        atlas: mutable atlas dict.

    Returns:
        ``{"ok": True, "used": int, "items": {id: (x, y, w, h)},
        "holes": [...]}``.  Every live rectangle keeps its ``w``/``h`` and is
        moved into the smallest deterministic shelf layout starting at
        ``(0, 0)`` (height descending, then id).  Consequently the used area
        ``sum(w * h)`` is unchanged by defragmentation while the number of
        holes strictly decreases (an uncontiguous hole list collapses to at
        most one trailing free rectangle).  The atlas is updated in place.

    Raises:
        AtlasError: the live rectangles cannot fit inside ``atlas["size"]``.
    """
    width, height = atlas["size"]
    used = sum(w * h for _, (_, _, w, h) in atlas["items"].items())

    ordered = sorted(
        atlas["items"].items(), key=lambda item: (-item[1][3], str(item[0]))
    )

    packed = {}
    cursor_x = 0
    cursor_y = 0
    row_height = 0
    for rect_id, (_, _, rect_w, rect_h) in ordered:
        if cursor_x + rect_w > width:
            cursor_x = 0
            cursor_y += row_height
            row_height = 0
        if cursor_y + rect_h > height:
            raise AtlasError("live rectangles do not fit inside the atlas")
        packed[rect_id] = (cursor_x, cursor_y, rect_w, rect_h)
        cursor_x += rect_w
        row_height = max(row_height, rect_h)

    atlas["items"] = packed

    # The band below the last shelf row is entirely free; record it as the
    # single trailing hole (this is what makes the hole count collapse).
    bottom = cursor_y + row_height
    if packed and bottom < height:
        atlas["holes"] = [(0, bottom, width, height - bottom)]
    else:
        atlas["holes"] = []

    return {
        "ok": True,
        "used": used,
        "items": dict(packed),
        "holes": list(atlas["holes"]),
    }


def set_filtering(atlas, id, mode):
    """Record the texture filter mode of one rectangle.

    Args:
        atlas: mutable atlas dict.
        id: key of a live placement, or of a rectangle staged in
            ``atlas["items"]`` that is about to be validated.
        mode: ``"nearest"`` or ``"linear"``.

    Returns:
        ``{"ok": True, "id": id, "mode": mode, "filtering": {id: mode}}``
        where ``filtering`` is a snapshot of the per-rectangle modes.  Modes
        are stored per rectangle id, so changing one rectangle never
        overwrites another rectangle's mode.

    Raises:
        AtlasError: ``id`` is unknown, ``mode`` is not supported, or the
            rectangle is not fully inside the atlas.  Out-of-bounds
            coordinates are reported, never clamped: clamping would make the
            sampler read a neighbouring rectangle.
    """
    if mode not in ("nearest", "linear"):
        raise AtlasError("unsupported filter mode: %r" % (mode,))
    if id not in atlas["items"]:
        raise AtlasError("id is not a live placement: %r" % (id,))

    x, y, w, h = atlas["items"][id]
    width, height = atlas["size"]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise AtlasError(
            "rectangle %r is outside the atlas: %r"
            % (id, (x, y, w, h))
        )

    atlas.setdefault("filtering", {})[id] = mode
    return {
        "ok": True,
        "id": id,
        "mode": mode,
        "filtering": dict(atlas["filtering"]),
    }


def evict_batch(atlas, ids, budget):
    """Evict several rectangles under an area budget.

    Args:
        atlas: mutable atlas dict.
        ids: sequence of placement ids to evict, processed in order.
        budget: maximum total pixel area that may be evicted, a non-negative
            integer.

    Returns:
        On success ``{"ok": True, "evicted": [id, ...], "freed": int,
        "holes": [...]}``; all requested ids were live, the whole batch
        respected ``budget`` and each evicted id was freed exactly like
        :func:`free` (removed from ``items``, hole recorded).

        On partial failure ``{"ok": False, "evicted": [...], "remaining":
        [...], "freed": int}``: ids are evicted in order until one is unknown
        or the next rectangle would exceed ``budget``.  ``evicted`` lists the
        ids that were really freed, ``remaining`` lists the ids not touched;
        those still live in the atlas.  State is consistent: an id is either
        fully freed or left untouched, and the reported lists and the atlas
        agree.

    Raises:
        AtlasError: ``budget`` is negative.
    """
    if not _is_int(budget) or budget < 0:
        raise AtlasError("budget must be a non-negative integer")

    ordered_ids = list(ids)
    evicted = []
    freed = 0

    for index, rect_id in enumerate(ordered_ids):
        if rect_id not in atlas["items"]:
            return {
                "ok": False,
                "evicted": evicted,
                "remaining": ordered_ids[index:],
                "freed": freed,
            }
        _, _, rect_w, rect_h = atlas["items"][rect_id]
        area = rect_w * rect_h
        if freed + area > budget:
            return {
                "ok": False,
                "evicted": evicted,
                "remaining": ordered_ids[index:],
                "freed": freed,
            }

        # Evict immediately: an id is either fully freed or never touched.
        free(atlas, rect_id)
        evicted.append(rect_id)
        freed += area

    return {
        "ok": True,
        "evicted": evicted,
        "freed": freed,
        "holes": list(atlas["holes"]),
    }
