#pragma once

#include <vector>
#include "neural_net_utility.hpp"

namespace neural_net
{
template < typename ActivationFunc_t >
class TrainingNeuron final
{
public:
    explicit TrainingNeuron( size_t t_connections ) : m_outConnections( t_connections ) {}

    void setOutput( float t_value ) { m_outputValue = t_value; }
    float getOutput() const { return m_outputValue; }
    float getOutgoingWeight( size_t t_index ) const { return m_outConnections[t_index].m_weight; }
    float getDeltaWeight( size_t t_index ) const { return m_outConnections[t_index].m_deltaWeight; }
    void setDeltaWeight( float t_value, size_t t_index ) { m_outConnections[t_index].m_deltaWeight = t_value; }
    size_t getNoOfConnections() const { return m_outConnections.size(); }

    template < typename Layer_t >
    void propogate( const Layer_t & t_previousLayer, size_t t_neuronId )
    {
        float sum( 0.f );
        for ( const TrainingNeuron & neuron : t_previousLayer )
        {
            sum += neuron.getOutput() * neuron.getOutgoingWeight( t_neuronId );
        }
        setOutput( ActivationFunc_t::activation( sum ) );
    }

    template < typename Input >
    Input getActivationFuncDerivative( Input t_input ) const
    {
        return static_cast< Input >( ActivationFunc_t::derivative( t_input ) );
    }

    void updateWeights()
    {
        for ( size_t i = 0; i < getNoOfConnections(); ++i )
        {
            m_outConnections[i].m_weight = m_outConnections[i].m_weight - 0.1 * m_outConnections[i].m_deltaWeight;
        }
    }

private:
    float m_outputValue;
    struct Connection
    {
        Connection() : m_weight( neural_net_utility::getRandomWeight() ), m_deltaWeight( 0 ) {}
        float m_weight;
        float m_deltaWeight;
    };
    std::vector< Connection > m_outConnections;
};

template < size_t NoOfInput_t,
           size_t HiddenLayersDimension_t,
           size_t NoOfHiddenLayers_t,
           size_t NoOfOutput_t,
           typename Neuron_t >
class NeuralNetwork final
{
public:
    using Layer_t = std::vector< Neuron_t >;

    NeuralNetwork()
    {
        Layer_t inputLayer;
        for ( size_t i = 0; i < NoOfInput_t; ++i )
        {
            inputLayer.push_back( Neuron_t( HiddenLayersDimension_t ) );
        }
        m_layers.push_back( inputLayer );

        for ( size_t i = 0; i < NoOfHiddenLayers_t; ++i )
        {
            Layer_t hiddenLayer;
            for ( size_t j = 0; j < HiddenLayersDimension_t; ++j )
            {
                if ( i == NoOfHiddenLayers_t - 1 )
                {
                    hiddenLayer.push_back( Neuron_t( NoOfOutput_t ) );
                }
                else
                {
                    hiddenLayer.push_back( Neuron_t( HiddenLayersDimension_t ) );
                }
            }
            m_layers.push_back( hiddenLayer );
        }

        Layer_t outputLayer;
        for ( size_t i = 0; i < NoOfOutput_t; ++i )
        {
            outputLayer.push_back( Neuron_t( 0 ) );
        }
        m_layers.push_back( outputLayer );
    }
    ~NeuralNetwork() = default;

    NeuralNetwork( const NeuralNetwork & ) = delete;
    NeuralNetwork( const NeuralNetwork && ) = delete;
    NeuralNetwork & operator=( NeuralNetwork & ) = delete;
    NeuralNetwork & operator=( NeuralNetwork && ) = delete;

    const std::vector< Layer_t > & getNet() const noexcept { return m_layers; }

    template < typename Data_t >
    void train( const Data_t & t_data )
    {
        for ( const auto & row : t_data )
        {
            for ( size_t i = 0; i < NoOfInput_t; ++i )
            {
                m_layers[0][i].setOutput( row[i] );
            }
            for ( size_t i = 1; i <= NoOfHiddenLayers_t + 1; ++i )
            {
                auto & currentLayer = m_layers[i];
                const Layer_t & previousLayer = m_layers[i - 1];
                for ( size_t j = 0; j < currentLayer.size(); ++j )
                {
                    currentLayer[j].propogate( previousLayer, j );
                }
            }

            backPropogation( row[2] );
        }
    }

private:
    template < typename T >
    void backPropogation( T t_desiredOutput )
    {
        const auto predictedOutput = m_layers.back().front().getOutput();
        const auto error = t_desiredOutput - predictedOutput;

        auto it = m_layers.begin() + NoOfHiddenLayers_t;
        auto & lastHiddenLayer = *it;

        for ( size_t i = 0; i < lastHiddenLayer.size(); ++i )
        {
            auto & neuron = lastHiddenLayer[i];
            const auto dEdW = -error * neuron.getOutput() * neuron.getActivationFuncDerivative( predictedOutput );
            neuron.setDeltaWeight( dEdW, 0 );
        }

        --it;
        for ( ; it >= m_layers.begin(); --it )
        {
            auto & currentLayer = *it;
            const auto & proceedingLayer = *( it + 1 );
            for ( size_t i = 0; i < currentLayer.size(); ++i )
            {
                auto & currentNeuron = currentLayer[i];
                for ( size_t j = 0; j < HiddenLayersDimension_t; j++ )
                {
                    float dEdout( 0 );
                    for ( size_t k = 0; k < proceedingLayer[j].getNoOfConnections(); ++k )
                    {
                        dEdout += proceedingLayer[j].getOutgoingWeight( k ) * proceedingLayer[j].getDeltaWeight( k );
                    }
                    const auto dEdW = dEdout * currentNeuron.getOutput() *
                                      currentNeuron.getActivationFuncDerivative( proceedingLayer[j].getOutput() );
                    currentNeuron.setDeltaWeight( dEdW, j );
                }
            }
        }

        for ( auto & layers : m_layers )
        {
            for ( auto & neuron : layers )
            {
                neuron.updateWeights();
            }
        }
    }

    std::vector< Layer_t > m_layers;
};
}
