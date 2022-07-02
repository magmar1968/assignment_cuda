#ifndef __MYTIMER__
#define __MYTIMER__

#include <chrono>
class Timer
{
public:
    Timer()
    {
        _StartTimepoint = std::chrono::high_resolution_clock::now();
    }
    ~Timer() {}

    void Stop()
    {
        using namespace std::chrono;
        auto endTimepoint = high_resolution_clock::now();

        auto start = time_point_cast<milliseconds>(_StartTimepoint).time_since_epoch().count();
        auto end = time_point_cast<milliseconds>(endTimepoint).time_since_epoch().count();

        auto ms = end - start;



        auto secs = ms / 1000;
        ms -= secs * 1000;
        auto mins = secs / 60;
        secs -= mins* 60;
        auto hour = mins / 60;
        mins -= hour * 60;

        std::cout << "Time performance:\n"
            << hour << " Hours : "
            << mins << " Minutes : "
            << secs << " Seconds : "
            << ms << " Milliseconds \n";

    }

private:
    std::chrono::time_point< std::chrono::high_resolution_clock> _StartTimepoint;
};

#endif
