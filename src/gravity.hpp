#ifndef __SourceTerms_HEADER__
#define __SourceTerms_HEADER__

#include "hydro.hpp"

class GravSourceTerms : public SourceTerms
{

public:
  std::valarray<double> AddSources(std::valarray<double> &P);

private:
  std::valarray<double> GetGravity(std::valarray<double> &dens);
} ;

#endif // __SourceTerms_HEADER__
