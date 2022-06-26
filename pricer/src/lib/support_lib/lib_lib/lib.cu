#include "lib.cuh"
namespace prcr
{
__host__ int 
countWords(std::string line)
{
    std::stringstream ss(line);
    int n_word = 0;
    std::string word; 
    while (ss >> word)
    {
        ++n_word;
    }
    return n_word;
}
    
}