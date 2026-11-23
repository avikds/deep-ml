def plan(program):
    temps = program["temps"]
    steps = program["steps"]

    n = len(steps)
    order = [step["name"] for step in steps]

    # Map each temporary to its producing step.
    producer = {}

    # Find the last step that reads each temporary.
    last_read = {}

    for i, step in enumerate(steps):
        out = step.get("out")
        if out is not None:
            producer[out] = i

        for name in step.get("ins", []):
            last_read[name] = i

    # Build inclusive lifetime intervals.
    intervals = {}

    for name, size in temps.items():
        start = producer.get(name, 0)
        end = last_read.get(name, start)

        intervals[name] = [start, end, size]

    # Scratch temporaries are live only during their own step.
    for i, step in enumerate(steps):
        scratch = step.get("scratch")

        if scratch is not None and scratch in temps:
            intervals[scratch] = [i, i, temps[scratch]]

    # Process temporaries in order of their starting time.
    names = sorted(
        intervals,
        key=lambda name: (
            intervals[name][0],
            -intervals[name][2],
            name
        )
    )

    offsets = {}
    active = []

    def align4(x):
        return (x + 3) & ~3

    def overlaps(a, b, c, d):
        return a < d and c < b

    def find_offset(size, current, excluded=None):
        """Find the lowest available aligned memory region."""

        occupied = []

        for name, off in active:
            if name == excluded:
                continue

            start, end, other_size = intervals[name]

            if start <= current <= end:
                occupied.append((off, off + other_size))

        occupied.sort()

        pos = 0

        for lo, hi in occupied:
            pos = align4(pos)

            if pos + size <= lo:
                return pos

            if pos < hi:
                pos = hi

        return align4(pos)

    for name in names:
        start, end, size = intervals[name]

        # Expire temporaries whose last use has already passed.
        active = [
            (other, off)
            for other, off in active
            if intervals[other][1] >= start
        ]

        assigned = False

        # Attempt legal in-place reuse.
        if name in producer:
            step_index = producer[name]
            step = steps[step_index]

            for inp in step.get("inplace", []):
                if inp not in offsets:
                    continue

                if last_read.get(inp, -1) != step_index:
                    continue

                candidate = offsets[inp]

                # Check whether the output can safely occupy
                # the input's region, including larger outputs.
                conflict = False

                for other, off in active:
                    if other == inp:
                        continue

                    other_size = intervals[other][2]

                    if overlaps(
                        candidate,
                        candidate + size,
                        off,
                        off + other_size
                    ):
                        conflict = True
                        break

                if not conflict:
                    offsets[name] = candidate
                    assigned = True
                    break

        # Otherwise allocate the lowest available memory region.
        if not assigned:
            offsets[name] = find_offset(size, start)

        active.append((name, offsets[name]))

    return order, offsets