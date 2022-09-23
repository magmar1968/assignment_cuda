#ifndef __STATISTIC_LIB__
#define __STATISTIC_LIB__


#define HD __host__ __device__
namespace prcr
{
    HD double avg(const double * array, const size_t dim);
    HD double sum_array(const double* array, const size_t dim);
    HD double dev_std(const double * arr, const size_t dim);
    HD double dev_std(const double * arr, const double mean  , const size_t dim);
    HD double dev_std(const double * arr, const double * arr2, const size_t dim);
    HD double compute_final_error(const double sq_sum, const double mean, size_t Ntot);
}









#endif
