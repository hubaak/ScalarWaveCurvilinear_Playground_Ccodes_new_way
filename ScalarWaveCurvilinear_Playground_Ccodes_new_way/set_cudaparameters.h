//const int Grid_size_x = gridDim.x;
//const int Grid_size_y = gridDim.y;
//const int Grid_size = Grid_size_x * Grid_size_y;
//const int Block_size_x = blockDim.x;
//const int Block_size_y = blockDim.y;
//const int Block_size = Block_size_x * Block_size_y;

const int NUM_threads = Grid_size * Block_size;

const int tx = threadIdx.x;
const int ty = threadIdx.y;
const int bx = blockIdx.x;
const int by = blockIdx.y;

const int blockid = by * Grid_size_x + bx;
const int tid = ty * Block_size_x + tx;
const int gid = blockid * Block_size + tid;



