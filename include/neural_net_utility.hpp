#pragma once

#include <math.h>

namespace neural_net_utility
{
namespace activation_function
{
    template < typename T >
    struct Tanh final
    {
        static T activation( T t_x ) { return tanh( t_x ); }
        static T derivative( T t_x ) { return 1.0 - t_x * t_x; }
    };

    template < typename T >
    struct Logistic final
    {
        static T activation( T t_x ) { return 1.0 / ( 1.0 + exp( -t_x ) ); }
        static T derivative( T t_x ) { return t_x * ( 1.0 - t_x ); }
    };
}

float getRandomWeight() { return static_cast< float >( rand() ) / RAND_MAX; }
}
