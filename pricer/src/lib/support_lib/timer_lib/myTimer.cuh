#ifndef __MYTIMER__
#define __MYTIMER__

#include <chrono>
#include <iostream>
namespace prcr
{

class Timer
{
public:
    Timer():
    _ms(0),_secs(0),_hour(0),_mins(0),_delta_ms(0)
    {
        _StartTimepoint = std::chrono::steady_clock::now();
    }
 
    ~Timer() {}

    void Stop()
    {

        _stopped = true;
        using namespace std::chrono;
        auto endTimepoint = steady_clock::now();
        auto duration = endTimepoint - _StartTimepoint;
        auto ms = duration_cast<milliseconds>(duration);
        
        _ms = ms.count();
        _delta_ms = ms.count();



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

    double GetDeltamsTime(){
        if(_stopped == true)
            return _delta_ms;
        else{
            Stop();
            return _delta_ms;
        }
    }

private:
    std::chrono::time_point< std::chrono::steady_clock> _StartTimepoint;
    int64_t _secs,_ms,_mins,_hour;
    int64_t _delta_ms;
    bool _stopped = false;
};
}

#endif