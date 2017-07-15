#include <iostream>

#include "data_reader.hpp"
#include "neural_net.hpp"
#include "neural_net_utility.hpp"

int main( int argc, char ** argv )
{
    if ( argc < 2 )
    {
        std::cerr << "Usage: " << argv[0] << " path_to_data" << std::endl;
        return 0;
    }

    using namespace neural_net;
    const auto & data = data_reader::read( argv[1] );
    NeuralNetwork< 2, 3, 1, 1, TrainingNeuron< neural_net_utility::activation_function::Tanh< double > > > nn;
    nn.train( data );

    return 1;
}
