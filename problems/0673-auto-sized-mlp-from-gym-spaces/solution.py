import math

def auto_mlp_from_spaces(obs_space: dict, act_space: dict, hidden_layers: list) -> dict:
    """
    Automatically determine MLP architecture from environment space definitions.
    """
    def get_dim(space):
        space_type = space["type"]

        if space_type == "Box":
            return math.prod(space["shape"])
        elif space_type in ("Discrete", "MultiBinary"):
            return int(space["n"])
        else:
            raise ValueError(f"Unsupported space type: {space_type}")

    input_dim = get_dim(obs_space)
    output_dim = get_dim(act_space)

    # Dimensions of all layers, including input and output.
    dimensions = [input_dim] + list(hidden_layers) + [output_dim]

    layer_shapes = []
    total_params = 0

    for in_features, out_features in zip(dimensions[:-1], dimensions[1:]):
        weight_shape = (in_features, out_features)
        bias_shape = (out_features,)

        layer_shapes.append((weight_shape, bias_shape))
        total_params += in_features * out_features + out_features

    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "layer_shapes": layer_shapes,
        "total_params": total_params,
    }