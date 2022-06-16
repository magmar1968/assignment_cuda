#include "statistic_lib.cuh"

__host__ __device__ double
average(const double * array, const size_t dim)
{
    double avg = 0.;
    for(int i = 0; i < dim; ++i)
        avg += array[i]/(double)dim;
    return avg;
}

__host__ __device__ double
dev_std(const double * array, const size_t dim)
{
    return dev_std(array,average(array,dim),dim);
}

__host__ __device__ double
dev_std(const double * array, const double mean, const size_t dim)
{
    double dev_std = 0.;
    for (int i = 0.; i < dim; ++i)
        dev_std += (array[i] - mean) * (array[i] - mean);
    
    return sqrt(dev_std/(double)dim);
}

__host__ __device__ double
dev_std(const double * array, const double * array2, const size_t dim)
{
    double avg = average(array,dim);
    double avg_2 = average(array2,dim);

    return sqrt(avg*avg - avg_2);//?
}