#include "statistic_lib.cuh"

namespace prcr
{
__host__ __device__ double
avg(const double * array, const size_t dim)
{
    double avg = 0.;
    for(int i = 0; i < dim; ++i)
        avg += array[i]/(double)dim;
    return avg;
}

__host__ __device__ double
dev_std(const double * array, const size_t dim)
{
    return dev_std(array,avg(array,dim),dim);
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
sum_array(const double* arr, const size_t dim)
{
    double sum_array = 0;
    for (int i = 0; i < dim; i++)
    {
        sum_array += arr[i];
    }
    return sum_array;
}

__host__ __device__ double
dev_std(const double * array, const double * array2, const size_t dim)
{
    double avg = prcr::avg(array,dim);
    double avg_2 = prcr::avg(array2,dim);

    return sqrt(avg*avg - avg_2);//? 
}

__host__ double
compute_final_error(const double sq_sum, const double mean, const size_t N_tot)
{
    return (sqrt((sq_sum/static_cast<double>(N_tot)-mean*mean)/N_tot));
}








}
