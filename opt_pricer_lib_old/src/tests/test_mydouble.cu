#include "../support_lib/myudouble.cuh"
#include "../support_lib/lib.cuh"
#include "../support_lib/myRandom/myRandom_gnr/combined.cuh"



__global__          void kernel();
__host__            void createUdoubleVec_host();
__device__          void createUdoubleVec_device();
__host__ __device__ void createUdoubleVec_generic();


__host__ __device__ void createUdoubleVec_generic(
                        uint * seeds,
                        size_t index,
                        size_t dim)
{

}

int main(int argc, char ** argv)
{
    using namespace pricer;
    srand(time(NULL));
    Device dev;
    if (cmdOptionExists(argv, argv + argc, "-h")) {
        std::cout << "usage: -[option] [attibute]       \n"
                  << "options: -h help                  \n"
                  << "         -g select gpu as device  \n"
                  << "         -c select cpu as device  \n"
                  << "         -f change output filename\n";
    }
    if ( cmdOptionExists(argv, argv + argc, "-cpu") or
         cmdOptionExists(argv, argv + argc, "-c"  ) or
        !cmdOptionExists(argv, argv + argc, "-gpu") or 
        !cmdOptionExists(argv, argv + argc, "-g"  )   )
        {
        dev.CPU = true;
    }
    if (cmdOptionExists(argv, argv + argc, "-gpu") || 
        cmdOptionExists(argv, argv + argc, "-g"  )    )
    {
        dev.GPU = true;
    }
    if (cmdOptionExists(argv, argv + argc, "-both"))
    {
        dev.GPU = true;
        dev.CPU = true;
    }


    

}