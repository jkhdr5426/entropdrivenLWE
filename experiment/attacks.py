def linearization_dimension(T, n):
    if T.__name__ == "T_identity":
        return n
    if T.__name__ == "T_quadratic":
        return n * (n + 1) // 2
    return n * 2
def linearization_dimension(T, n):
    if T.__name__ == "T_identity":
        return n
    if T.__name__ == "T_quadratic":
        return n * (n + 1) // 2
    return n * 2

# The effective attack dimension grows quadratically