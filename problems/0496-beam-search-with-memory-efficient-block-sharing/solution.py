import numpy as np
import math


def beam_search_block_sharing(log_probs, beam_width, block_size, eos_token=-1):
    """
    Perform beam search decoding with memory-efficient block sharing.

    Args:
        log_probs: numpy array of shape (max_steps, vocab_size)
        beam_width: number of beams to maintain
        block_size: number of tokens per memory block
        eos_token: end-of-sequence token id (-1 for no early stopping)

    Returns:
        dict with keys:
            'sequences'
            'scores'
            'total_blocks_allocated'
            'blocks_in_use_final'
            'naive_blocks_needed'
    """

    log_probs = np.asarray(log_probs, dtype=float)

    if log_probs.ndim != 2:
        raise ValueError("log_probs must be a 2D array")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    max_steps, vocab_size = log_probs.shape

    # ------------------------------------------------------------
    # Physical block store
    # ------------------------------------------------------------
    blocks = {}       # block_id -> list of tokens
    ref_count = {}    # block_id -> number of beam references
    next_block_id = 0
    total_blocks_allocated = 0

    def allocate_block(tokens):
        nonlocal next_block_id, total_blocks_allocated

        block_id = next_block_id
        next_block_id += 1
        total_blocks_allocated += 1

        blocks[block_id] = list(tokens)
        ref_count[block_id] = 1

        return block_id

    def incref(block_id):
        ref_count[block_id] += 1

    def decref(block_id):
        ref_count[block_id] -= 1

        if ref_count[block_id] == 0:
            del ref_count[block_id]
            del blocks[block_id]

    def release_beam(beam):
        for block_id in beam["block_table"]:
            decref(block_id)

    def append_to_shared_beam(parent, token, direct_ownership=False):
        """
        Create a child by appending token.

        If direct_ownership=True, transfer the parent's block-table
        ownership to the child. This is the 'last child' optimization.

        Otherwise, the child initially shares all parent blocks and
        performs copy-on-write if the final block is not full.
        """

        table = list(parent["block_table"])

        if direct_ownership:
            # Parent's references are transferred to this child.
            # No incref needed.
            pass
        else:
            # Child shares all existing blocks with parent.
            for block_id in table:
                incref(block_id)

        # If no block exists yet, allocate one.
        if not table:
            table.append(allocate_block([token]))
            return table

        last_id = table[-1]

        # There is room in the last block.
        if len(blocks[last_id]) < block_size:
            if ref_count[last_id] > 1:
                # Copy-on-write: detach this child from the shared block.
                old_tokens = list(blocks[last_id])

                decref(last_id)

                new_id = allocate_block(old_tokens + [token])
                table[-1] = new_id
            else:
                # Sole owner: mutate in place.
                blocks[last_id].append(token)

        else:
            # Last block is full: allocate a new physical block.
            table.append(allocate_block([token]))

        return table

    def materialize_sequence(beam):
        tokens = []
        for block_id in beam["block_table"]:
            tokens.extend(blocks[block_id])
        return tokens[:beam["length"]]

    # ------------------------------------------------------------
    # Step 0: initialize beams from top tokens
    # ------------------------------------------------------------
    if max_steps == 0:
        return {
            "sequences": [],
            "scores": [],
            "total_blocks_allocated": 0,
            "blocks_in_use_final": 0,
            "naive_blocks_needed": 0
        }

    initial_order = np.argsort(-log_probs[0], kind="stable")
    num_initial = min(beam_width, vocab_size)

    beams = []

    for token in initial_order[:num_initial]:
        token = int(token)

        block_id = allocate_block([token])

        beams.append({
            "block_table": [block_id],
            "length": 1,
            "score": float(log_probs[0, token]),
            "finished": (eos_token != -1 and token == eos_token),
        })

    # ------------------------------------------------------------
    # Subsequent decoding steps
    # ------------------------------------------------------------
    for t in range(1, max_steps):

        # If every beam has reached EOS, stop.
        if eos_token != -1 and all(beam["finished"] for beam in beams):
            break

        candidates = []

        for parent_idx, parent in enumerate(beams):

            # Finished beams are carried forward unchanged.
            if parent["finished"]:
                candidates.append({
                    "parent_idx": parent_idx,
                    "token": None,
                    "score": parent["score"],
                    "finished": True,
                    "is_carry": True,
                })
                continue

            # Expand active beam with every vocabulary token.
            for token in range(vocab_size):
                candidates.append({
                    "parent_idx": parent_idx,
                    "token": token,
                    "score": parent["score"] + float(log_probs[t, token]),
                    "finished": (
                        eos_token != -1 and token == eos_token
                    ),
                    "is_carry": False,
                })

        # Top beam_width candidates, stable tie-breaking.
        scores = np.array([c["score"] for c in candidates])
        order = np.argsort(-scores, kind="stable")
        selected = [candidates[i] for i in order[:beam_width]]

        # --------------------------------------------------------
        # Determine how many selected children each parent has.
        # This lets the final child reuse the parent's block table
        # directly, while earlier siblings use COW.
        # --------------------------------------------------------
        selected_by_parent = {}

        for rank, candidate in enumerate(selected):
            if candidate["is_carry"]:
                continue

            p = candidate["parent_idx"]

            selected_by_parent.setdefault(p, [])
            selected_by_parent[p].append(rank)

        last_selected_for_parent = {
            p: ranks[-1]
            for p, ranks in selected_by_parent.items()
        }

        new_beams = []

        # Track which old parents transfer ownership or get released.
        transferred_parents = set()

        for rank, candidate in enumerate(selected):
            parent_idx = candidate["parent_idx"]
            parent = beams[parent_idx]

            # A finished beam that survives simply keeps its ownership.
            if candidate["is_carry"]:
                new_beams.append({
                    "block_table": parent["block_table"],
                    "length": parent["length"],
                    "score": parent["score"],
                    "finished": True,
                })

                transferred_parents.add(parent_idx)
                continue

            token = int(candidate["token"])

            # The last selected child from a parent takes direct ownership.
            direct = (last_selected_for_parent[parent_idx] == rank)

            child_table = append_to_shared_beam(
                parent,
                token,
                direct_ownership=direct
            )

            new_beams.append({
                "block_table": child_table,
                "length": parent["length"] + 1,
                "score": candidate["score"],
                "finished": candidate["finished"],
            })

            if direct:
                transferred_parents.add(parent_idx)

        # --------------------------------------------------------
        # Release old parent references that were not transferred.
        # --------------------------------------------------------
        for parent_idx, parent in enumerate(beams):
            if parent_idx not in transferred_parents:
                release_beam(parent)

        beams = new_beams

    # ------------------------------------------------------------
    # Final ordering by descending score
    # ------------------------------------------------------------
    beams.sort(key=lambda b: b["score"], reverse=True)

    sequences = [
        materialize_sequence(beam)
        for beam in beams[:beam_width]
    ]

    scores = [
        round(float(beam["score"]), 4)
        for beam in beams[:beam_width]
    ]

    # Distinct blocks referenced by the final beams.
    final_block_ids = set()

    for beam in beams[:beam_width]:
        final_block_ids.update(beam["block_table"])

    blocks_in_use_final = len(final_block_ids)

    # Naive independent allocation.
    naive_blocks_needed = (
        beam_width * math.ceil(max_steps / block_size)
    )

    return {
        "sequences": sequences,
        "scores": scores,
        "total_blocks_allocated": total_blocks_allocated,
        "blocks_in_use_final": blocks_in_use_final,
        "naive_blocks_needed": naive_blocks_needed,
    }