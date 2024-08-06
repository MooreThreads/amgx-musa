/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <musa.h>

namespace cusp
{
namespace detail
{

class timer
{
  public:
    size_t calls;
    bool paused;
    double milliseconds;
    musaEvent_t _start;
    musaEvent_t _end;

    timer() : _start(NULL), _end(NULL)
    { 
      reset();
    }

    ~timer()
    {
      stop();
    }

    void operator+=(const timer &b) 
    {
      milliseconds += b.milliseconds;
      calls += b.calls;
    }

    bool is_empty(void)  const { return milliseconds == 0.0; }
    bool is_paused(void) const { return paused; }

    void unpause(void) 
    { 
      musaEventRecord(_start,0);
      paused = false; 
    }

    void pause(void) 
    { 
      stop();
      musaEventCreate(&_start); 
      musaEventCreate(&_end);
      paused = true; 
    }            

    void start(void) 
    { 
      ++calls; 
      musaEventCreate(&_start); 
      musaEventCreate(&_end);
      musaEventRecord(_start,0);
    }

    void stop(void) 
    { 
      if(_start != NULL && _end != NULL )
      {
        milliseconds += milliseconds_elapsed(); 

        musaEventDestroy(_start);
        musaEventDestroy(_end);

        _start = NULL;
        _end   = NULL;
      }
    }

    void soft_stop(void) 
    { 
      if ( !paused ) 
        milliseconds = milliseconds_elapsed(); 
    }

    void reset(void) 
    { 
      if(_start != NULL ) musaEventDestroy(_start);
      if(_end   != NULL ) musaEventDestroy(_end);

      musaEventCreate(&_start); 
      musaEventCreate(&_end);

      calls = 0;
      paused = false;
      milliseconds = 0.0;
    }

    void soft_reset(void) 
    { 
      calls = 0; 
      milliseconds = 0.0; 

      musaEventDestroy(_start);
      musaEventDestroy(_end);
      musaEventCreate(&_start); 
      musaEventCreate(&_end);
    }

    float milliseconds_elapsed()
    { 
      float elapsed_time;
      musaEventRecord(_end, 0);
      musaEventSynchronize(_end);
      musaEventElapsedTime(&elapsed_time, _start, _end);
      return elapsed_time;
    }

    float seconds_elapsed()
    { 
      return milliseconds_elapsed() / 1000.0;
    }

};

} // end namespace detail
} // end namespace cusp

