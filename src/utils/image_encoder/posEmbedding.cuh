
#define EPSILON 1e-6
#define TEMPERATURE 10000
#define NUM_POS_FEATS 50

template<typename T, typename accFloatT>
__global__ void posEmbeddingKernel(T d_input[Ni][Ny][Nx], 
                                   T d_output[Ni][Ny][Nx],
                                   T d_dimensions_x[Nx],
                                   T d_dimensions_y[Ny],
                                   int numPosFeats);
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y ;
    int n = threadIdx.z + blockIdx.z * blockDim.z;

    accFloatT y_embed = (y+1)/(Ny+EPSILON);
    accFloatT x_embed = (x+1)/(Nx+EPSILON);

    T pos_embeds_x[numPosFeats];
    T pos_embeds_y[numPosFeats];
    for (int i = 0; i < numPosFeats; i++)
    {
        pos_embeds_y[i] = y_embed / d_dimensions_y[i];
        pos_embeds_x[i] = x_embed / d_dimensions_x[i];
    }

    



}