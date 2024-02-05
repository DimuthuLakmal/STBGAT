def organize_matrix(mat):
    """
    Reduce the input x dimenstion from 4 to 3
    Parameters
    ----------
    mat: Tensor, input x to encoder, size of B, T, N, F

    Returns
    -------
    mat: Tensor, input x, reshaped to B * N, T, F
    """
    mat = mat.permute(1, 0, 2, 3)  # B, T, N, F -> T, B, N , F (4, 36, 170, 16) -> (36, 4, 170, 16)
    mat_shp = mat.shape
    mat = mat.reshape(mat_shp[0], mat_shp[1] * mat_shp[2], mat_shp[3])  # (36, 4 * 170, 16)
    mat = mat.permute(1, 0, 2)  # (4 * 170, 36, 16)
    return mat