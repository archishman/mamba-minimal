array axpby(
    const array& x, // Input array x
    const array& y, // Input array y
    const float alpha, // Scaling factor for x
    const float beta, // Scaling factor for y
    StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
) {
    // Scale x and y on the provided stream
    auto ax = multiply(array(alpha), x, s);
    auto by = multiply(array(beta), y, s);

    // Add and return
    return add(ax, by, s);
}