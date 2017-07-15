#pragma once

#include <string>
#include <vector>
#include <fstream>

namespace data_reader
{
using data_t = std::vector< std::vector< float > >;

data_t read( const std::string & t_dataSetPath )
{
    data_t data;
    std::ifstream file( t_dataSetPath );
    while ( !file.eof() )
    {
        std::string line;
        std::getline( file, line );
        if ( !line.empty() )
        {
            std::vector< float > row;
            auto delimIndx = line.find( "," );
            while ( delimIndx != std::string::npos )
            {
                row.push_back( std::stof( line.substr( 0, delimIndx ) ) );
                line = line.substr( delimIndx + 1 );
                delimIndx = line.find( "," );
            }
            row.push_back( std::stof( line ) );
            data.push_back( row );
        }
    }

    return data;
}
}
