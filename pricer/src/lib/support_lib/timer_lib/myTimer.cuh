#ifndef __MYTIMER__
#define __MYTIMER__

#include <chrono>
#include <iostream>
namespace prcr
{

class Timer
{
public:
    Timer()
    {
        _StartTimepoint = std::chrono::steady_clock::now();
    }
 
    ~Timer() {}

    void Stop()
    {

        _stopped = true;
        using namespace std::chrono;
        auto endTimepoint = steady_clock::now();

        auto start = time_point_cast<milliseconds>(_StartTimepoint).time_since_epoch().count();
        auto end = time_point_cast<milliseconds>(endTimepoint).time_since_epoch().count();

        _ms = end - start;



        _secs   = _ms   / 1000;
        _ms    -= _secs * 1000;
        _mins   = _secs / 60;
        _secs  -= _mins * 60;
        _hour   = _mins / 60;
        _mins  -= _hour * 60;


    }
    
    void PrintTime()
    {
        std::cout << "Time performance:\n"
            << _hour << " Hours : "
            << _mins << " Minutes : "
            << _secs << " Seconds : "
            << _ms << " Milliseconds \n";

    } 

    double GetTime(){
        if(_stopped == true)
            return _ms;
        else{
            Stop();
            return _ms;
        }
    }

private:
    std::chrono::time_point< std::chrono::steady_clock> _StartTimepoint;
    double _secs,_ms,_mins,_hour;
    bool _stopped = false;
};
}

#endif
