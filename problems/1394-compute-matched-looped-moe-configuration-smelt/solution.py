import math


def block_flops(d, d_ff, top_k):
    return 4 * d * d + top_k * 3 * d * d_ff


def block_params(d, d_ff, n_experts):
    return 4 * d * d + n_experts * 3 * d * d_ff


def smelt_match(d, n_blocks, n_experts, top_k, ff_mult, kv_heads, head_dim, loop_fraction=0.5):
    lam = 1.0 + loop_fraction
    applications = round(n_blocks * lam)

    # Narrow width and round down to a multiple of 8.
    d_new = int(d / math.sqrt(lam))
    d_new = (d_new // 8) * 8
    d_ff_new = ff_mult * d_new

    # Base total parameter count.
    base_params = n_blocks * block_params(
        d, ff_mult * d, n_experts
    )

    # Find the smallest expert count meeting the parameter target.
    experts_new = 1
    while (
        n_blocks * block_params(d_new, d_ff_new, experts_new)
        < base_params
    ):
        experts_new += 1

    # Reduce KV heads to compensate for repeated block applications.
    kv_heads_new = max(1, math.floor(kv_heads / lam + 0.5))

    base_flops = n_blocks * block_flops(
        d, ff_mult * d, top_k
    )
    looped_flops = applications * block_flops(
        d_new, d_ff_new, top_k
    )

    looped_params = n_blocks * block_params(
        d_new, d_ff_new, experts_new
    )

    # Include the application-count multiplier in the KV ratio.
    kv_ratio = (
        applications * kv_heads_new
        / (n_blocks * kv_heads)
    )

    return {
        "d_new": d_new,
        "d_ff_new": d_ff_new,
        "experts_new": experts_new,
        "kv_heads_new": kv_heads_new,
        "applications": applications,
        "flops_ratio": round(looped_flops / base_flops, 4),
        "params_ratio": round(looped_params / base_params, 4),
        "kv_ratio": round(kv_ratio, 4),
    }