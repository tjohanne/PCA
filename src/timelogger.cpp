#include "include/timelogger.h"
TimeLogger::TimeLogger(int features, int samples, int n_components, std::string log_name) : 
    features{features}, samples{samples}, n_components{n_components}, log_name{log_name} {}
TimeLogger::timeLog* TimeLogger::start(std::string name) {
    timeLog *tl = new timeLog(this->features, this->samples, this->n_components, name, std::chrono::high_resolution_clock::now());
    logs.push_back(tl);
    return tl;
}
void TimeLogger::stop(timeLog* tl) {
    auto end = std::chrono::high_resolution_clock::now();
    tl->end_time = end;
    auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - tl->start_time);
    tl->time_ms = elapsed.count() * 1e-9;
}


