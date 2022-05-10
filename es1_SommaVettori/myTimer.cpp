#include "myTimer.h"

Timer::Timer()
{

    m_StartTimepoint = std::chrono::high_resolution_clock::now();
}

Timer::~Timer()
{
    Stop();
}

void Timer::Stop()
{
    using namespace std::chrono;
    auto endTimepoint = high_resolution_clock::now();

    auto start = time_point_cast<milliseconds>(m_StartTimepoint).time_since_epoch().count();
    auto end   = time_point_cast<milliseconds>(endTimepoint).time_since_epoch().count();

    auto ms = end - start;


   
    auto secs = ms/1000;
    ms -=secs * 1000;
    auto mins = secs / 60;
    secs -=  secs * 60;
    auto hour = mins / 60;
    mins -= hour * 60;

    std::cout << "Time performance:\n"
              << hour << " Hours : " 
              << mins << " Minutes : " 
              << secs << " Seconds : " 
              << ms   << " Milliseconds \n";
    
}