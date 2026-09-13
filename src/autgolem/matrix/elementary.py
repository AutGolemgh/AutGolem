import torch

def rowswap(matrix, source_row, target_row):
    new_matrix = matrix.clone()
    new_matrix[[source_row, target_row]] = new_matrix[[target_row, source_row]]
    return new_matrix

def rowscale(matrix, source_row, scaling_factor):
    new_matrix = matrix.clone()
    new_matrix[source_row] = new_matrix[source_row] * scaling_factor
    return new_matrix

def rowreplacement(matrix, row_i, row_j, j, k):
    scaled_i = rowscale(matrix, row_i, j)
    scaled_j = rowscale(matrix, row_j, k)
    new_matrix = matrix.clone()
    new_matrix[row_i] = scaled_i[row_i] + scaled_j[row_j]
    return new_matrix

def rref(matrix):
    """
    Computes the Reduced Row Echelon Form (RREF) of a matrix using
    Gauss-Jordan elimination.

    The function goes through the matrix column by column. For each
    column, it looks at the current pivot position. If that position
    is zero, it searches the rows below for a non-zero value and swaps
    that row into place using rowswap. If the whole column is zero, it
    skips to the next column.

    Once a non-zero pivot is found, it is scaled to 1 using rowscale.
    Then, all other rows in the matrix (not just the ones below) are
    updated using rowreplacement, so that every other entry in that
    column becomes 0. This ensures the result is fully reduced, not
    just in row echelon form.

    The pivot row counter only increases when a pivot is successfully
    placed. The process stops once every row has a pivot.
    """
    new_matrix = matrix.clone()
    rows, cols = new_matrix.shape
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        if new_matrix[pivot_row, col] == 0:
            nonzero_row = None
            for r in range(pivot_row + 1, rows):
                if new_matrix[r, col] !=0:
                    nonzero_row = r
                    break
            if nonzero_row is None:
                continue
            new_matrix = rowswap(new_matrix, pivot_row, nonzero_row)

        if new_matrix[pivot_row, col] != 0:
            pivot_val = new_matrix[pivot_row, col].item()
            new_matrix = rowscale(new_matrix, pivot_row, 1.0/pivot_val)

        for i in range(rows):
            if i != pivot_row and new_matrix[i, col] != 0:
                factor = new_matrix[i, col].item()
                new_matrix = rowreplacement(new_matrix, i, pivot_row, 1.0, -factor)

        pivot_row += 1
    return new_matrix