__kernel void hello(
    __global const unsigned int *a,
    __global const unsigned int *b,
    __global unsigned int *c
)
{
    size_t gid = get_global_id(1) * 1024 + get_global_id(0);
    //size_t gid = mad24(get_global_id(1), (size_t)1024, get_global_id(0));

    uint temp_a = a[gid];
    uint temp_b = b[gid];

    uint temp_c;
    temp_c  = ((temp_a & 0x000000FF) + (temp_b & 0x000000FF)) & 0x000000FF;
    temp_c |= ((temp_a & 0x0000FF00) + (temp_b & 0x0000FF00)) & 0x0000FF00;
    temp_c |= ((temp_a & 0x00FF0000) + (temp_b & 0x00FF0000)) & 0x00FF0000;
    temp_c |= ((temp_a & 0xFF000000) + (temp_b & 0xFF000000)) & 0xFF000000;

    c[gid] = temp_c;
}
