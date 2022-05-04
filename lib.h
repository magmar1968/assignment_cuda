#include <algorithm>
#include <string>
#include <random>
#include <chrono>

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

std::string getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

void fillArray(float * a, int size, std::default_random_engine eng)
{
    std::uniform_real_distribution<> urd(-10, 10);

    for(int i = 0; i < size; ++i)
    {
        a[i] = urd(eng);
    }
}

class Timer
{
  public:
    Timer()
    {
        _StartTimepoint = std::chrono::high_resolution_clock::now();
    }
    ~Timer(){}

    double getTimeDiff()
    {
        using namespace std::chrono;
        auto endTimepoint = high_resolution_clock::now();
        auto start = time_point_cast<milliseconds>(_StartTimepoint).time_since_epoch().count();
        auto end   = time_point_cast<milliseconds>(endTimepoint).time_since_epoch().count();

        auto ms = end - start;
        return ms;
    }

  private:
    std::chrono::time_point< std::chrono::high_resolution_clock> _StartTimepoint;
};



