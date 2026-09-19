import numpy as np

def pca_color_augmentation(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Apply PCA color augmentation to an RGB image.
    """
    img = np.asarray(image, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)

    pixels = img.reshape(-1, 3)

    # With zero distortion, return the original image exactly.
    if np.all(alpha == 0):
        return img.copy()

    mean = np.mean(pixels, axis=0)
    centered = pixels - mean

    # Covariance of RGB channels.
    cov = np.cov(centered, rowvar=False)

    # Numerical symmetry.
    cov = 0.5 * (cov + cov.T)

    # Eigen-decomposition as required.
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by descending eigenvalue.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Protect against tiny negative eigenvalues from floating-point error.
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # AlexNet PCA color distortion.
    distortion = eigenvectors @ (
        alpha * np.sqrt(eigenvalues)
    )

    augmented = pixels + distortion

    # Clamp RGB values.
    augmented = np.clip(augmented, 0.0, 255.0)

    return augmented.reshape(img.shape)