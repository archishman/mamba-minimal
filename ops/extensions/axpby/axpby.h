/**
*  Scale and sum two vectors element-wise
*  z = alpha * x + beta * y
*
*  Follow numpy style broadcasting between x and y
*  Inputs are upcasted to floats if needed
**/
array axpby(
    const array& x, // Input array x
    const array& y, // Input array y
    const float alpha, // Scaling factor for x
    const float beta, // Scaling factor for y
    StreamOrDevice s = {} // Stream on which to schedule the operation
);