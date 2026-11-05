import numpy as np
from sklearn.cluster import MiniBatchKMeans


def _get_patches(images):
    n = len(images)
    return (
        images.reshape(n, 8, 8, 8, 8)
        .transpose(0, 1, 3, 2, 4)
        .reshape(-1, 64)
        .astype(np.float32)
    )


def _quantize_centers(centers):
    maximum = float(np.max(np.abs(centers)))
    scale = np.float32(max(maximum / 127.0, 1e-8))

    q = np.clip(
        np.rint(centers / scale + 128.0),
        0,
        255
    ).astype(np.uint8)

    reconstructed = (q.astype(np.float32) - 128.0) * scale

    return q, scale, reconstructed


def fit(train_images):
    """
    Learn four additive codebooks for 8x8 image patches.

    State size:
        4 * 256 * 64 = 65,536 bytes.

    The last entry of each codebook stores its float32 scale.
    The remaining 255 entries are actual codewords.
    """
    X = np.asarray(train_images, dtype=np.float32)

    patches = _get_patches(X)
    residual = patches.copy()

    books = np.zeros((4, 256, 64), dtype=np.uint8)

    for stage in range(4):
        km = MiniBatchKMeans(
            n_clusters=255,
            batch_size=2048,
            max_iter=35,
            n_init=1,
            random_state=42 + stage,
            reassignment_ratio=0.01,
            max_no_improvement=10,
            compute_labels=False
        )

        km.fit(residual)

        centers = km.cluster_centers_.astype(np.float32)

        # Quantize dictionary entries to signed 8-bit values
        # represented through uint8 with offset 128.
        q, scale, decoded_centers = _quantize_centers(centers)

        books[stage, :255] = q

        # Store each stage's scale in its reserved 256th entry.
        scale_bytes = np.frombuffer(
            np.float32(scale).tobytes(),
            dtype=np.uint8
        )
        books[stage, 255, :4] = scale_bytes

        # Subtract exactly what the decoder will reconstruct.
        norms = np.sum(decoded_centers ** 2, axis=1)
        distances = (
            np.sum(residual ** 2, axis=1, keepdims=True)
            + norms[None, :]
            - 2.0 * residual @ decoded_centers.T
        )

        indices = np.argmin(distances, axis=1)
        residual -= decoded_centers[indices]

    return books.tobytes()


def _load_books(state):
    books = np.frombuffer(
        state,
        dtype=np.uint8
    ).reshape(4, 256, 64)

    centers = []
    scales = []

    for stage in range(4):
        scale = np.frombuffer(
            books[stage, 255, :4].tobytes(),
            dtype=np.float32
        )[0]

        decoded = (
            books[stage, :255].astype(np.float32) - 128.0
        ) * scale

        centers.append(decoded)
        scales.append(scale)

    return centers


def encode(state, image):
    """
    Encode one face into exactly 256 bytes.

    Each of the 64 patches receives four 8-bit code indices.
    """
    image = np.asarray(image, dtype=np.float32)

    patches = (
        image.reshape(8, 8, 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(64, 64)
        .copy()
    )

    centers = _load_books(state)

    codes = np.empty((64, 4), dtype=np.uint8)

    for stage in range(4):
        dictionary = centers[stage]

        norms = np.sum(dictionary ** 2, axis=1)

        distances = (
            np.sum(patches ** 2, axis=1, keepdims=True)
            + norms[None, :]
            - 2.0 * patches @ dictionary.T
        )

        indices = np.argmin(distances, axis=1)

        codes[:, stage] = indices.astype(np.uint8)

        patches -= dictionary[indices]

    return codes.tobytes()


def decode(state, bitstream):
    """
    Reconstruct one 64x64 face using its four-stage
    additive vector-quantization representation.
    """
    codes = np.frombuffer(
        bitstream,
        dtype=np.uint8
    ).reshape(64, 4)

    centers = _load_books(state)

    patches = np.zeros((64, 64), dtype=np.float32)

    for stage in range(4):
        patches += centers[stage][codes[:, stage]]

    image = (
        patches.reshape(8, 8, 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(64, 64)
    )

    return np.clip(image, 0.0, 1.0)