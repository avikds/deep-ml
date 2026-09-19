def analyze_ml_pipeline(tasks: list) -> dict:
    if not tasks:
        return {
            "execution_order": [],
            "earliest_start": {},
            "earliest_finish": {},
            "latest_start": {},
            "latest_finish": {},
            "slack": {},
            "critical_path": [],
            "makespan": 0
        }

    task_map = {task["id"]: task for task in tasks}
    ids = sorted(task_map.keys())

    # Build graph and indegree
    dependents = {tid: [] for tid in ids}
    indegree = {tid: 0 for tid in ids}

    for task in tasks:
        tid = task["id"]
        for dep in task["dependencies"]:
            dependents[dep].append(tid)
            indegree[tid] += 1

    # Topological sort with alphabetical tie-breaking
    available = sorted(
        tid for tid in ids
        if indegree[tid] == 0
    )

    execution_order = []

    while available:
        tid = available.pop(0)
        execution_order.append(tid)

        for nxt in sorted(dependents[tid]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                available.append(nxt)

        available.sort()

    # Forward pass
    earliest_start = {}
    earliest_finish = {}

    for tid in execution_order:
        deps = task_map[tid]["dependencies"]

        if deps:
            earliest_start[tid] = max(
                earliest_finish[d] for d in deps
            )
        else:
            earliest_start[tid] = 0

        earliest_finish[tid] = (
            earliest_start[tid] + task_map[tid]["duration"]
        )

    makespan = max(earliest_finish.values())

    # Backward pass
    latest_finish = {}
    latest_start = {}

    for tid in reversed(execution_order):
        successors = dependents[tid]

        if successors:
            latest_finish[tid] = min(
                latest_start[s] for s in successors
            )
        else:
            latest_finish[tid] = makespan

        latest_start[tid] = (
            latest_finish[tid] - task_map[tid]["duration"]
        )

    # Slack
    slack = {
        tid: latest_start[tid] - earliest_start[tid]
        for tid in execution_order
    }

    # Critical path: zero-slack tasks in execution order
    critical_path = [
        tid for tid in execution_order
        if slack[tid] == 0
    ]

    return {
        "execution_order": execution_order,
        "earliest_start": {
            tid: earliest_start[tid] for tid in execution_order
        },
        "earliest_finish": {
            tid: earliest_finish[tid] for tid in execution_order
        },
        "latest_start": {
            tid: latest_start[tid] for tid in execution_order
        },
        "latest_finish": {
            tid: latest_finish[tid] for tid in execution_order
        },
        "slack": {
            tid: slack[tid] for tid in execution_order
        },
        "critical_path": critical_path,
        "makespan": makespan
    }