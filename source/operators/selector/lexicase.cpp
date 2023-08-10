// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <numeric>
#include "operon/operators/selector.hpp"
#include "operon/core/dataset.hpp"
#include "operon/error_metrics/mean_absolute_error.hpp"


namespace Operon {

// Semi-Dynamic e-Lexicase selection as in https://doi.org/10.1162/evco_a_00224
auto LexicaseSelector::operator()(Operon::RandomGenerator& random) const -> size_t
{   
    // **indexes** of individuals
    pool_.resize(Population().size());
    std::iota(pool_.begin(), pool_.end(), 0);

    testCases_.resize(range_.Size());
    std::iota(testCases_.begin(), testCases_.end(), range_.Start());
    std::shuffle(testCases_.begin(), testCases_.end(), random);

    // vector of individuals to use in the round
    std::vector<Individual> lexiIndividuals;
    
    // shuffled fitness cases indexes. going to pop one at a time
    while (testCases_.size()>0 && pool_.size()>1) {
        size_t caseIdx = testCases_.back();

        lexiIndividuals.resize(0);
        std::transform(pool_.begin(), pool_.end(), std::back_inserter(lexiIndividuals),
            [this, pop=Population(), caseIdx, &random](size_t idx) {
                Individual newIndividual(pop[idx].Size());

                newIndividual.Genotype = pop[idx].Genotype;
                newIndividual.Fitness = GetEvaluator().SingleLexiCase(random, newIndividual, caseIdx);

                return newIndividual;
            }
        );

        Operon::Span<Individual const> pop = { lexiIndividuals.data(), lexiIndividuals.size()};

        // Finds the `elite` fitness in pop based on case (x_i, y_i)_{i=caseIdx}, based on comp_
        auto elite = *std::min_element(pop.begin(), pop.end(),
            [this](Individual a, Individual b){ return Compare(a, b); }); 
        
        // estimation of variability of pool using the Median Absolute Deviation (MAD)
        std::vector<Operon::Scalar> MAD = MadFitnesses(pop);

        // Filter pool, keeping only non-dominated candidate indexes
        size_t index = 0;
        std::erase_if(pool_, [this, pop, elite, MAD, &index](size_t originalIdx){ 
            return IsNonDominated(pop[index++], elite, MAD);
        });

        testCases_.pop_back();
    }
    
    return pool_[ std::uniform_int_distribution<size_t>(0, pool_.size() - 1)(random) ];
}

void LexicaseSelector::Prepare(const Operon::Span<const Individual> pop) const
{
    SelectorBase::Prepare(pop);
}

bool LexicaseSelector::IsNonDominated(Individual const& lhs, Individual const& rhs, std::vector<Operon::Scalar> eps) const
{
    EXPECT(std::size(lhs.Fitness) == std::size(rhs.Fitness));
    
    auto const& fit1 = lhs.Fitness;
    auto const& fit2 = rhs.Fitness;

    // TODO: overload comparisons to take as eps an Vector<scalar> instead of a single value
    auto d = ParetoDominance{}(fit1.begin(), fit1.end(), fit2.begin(), fit2.end(), eps.begin(), eps.end());
    
    bool accept = ( (d!=Dominance::None) && (d!=Dominance::Equal) );

    return accept;
}

Operon::Scalar LexicaseSelector::Median(const Operon::Span<Operon::Scalar>& v) const
{
    std::vector<Operon::Scalar> x(v.size());
    x.assign(v.data(),v.data()+v.size());

    // middle element
    size_t n = x.size()/2;

    // sort nth element of array
    std::nth_element(x.begin(),x.begin()+n,x.end());

    // if evenly sized, return average of middle two elements
    if (x.size() % 2 == 0) {
        std::nth_element(x.begin(),x.begin()+n-1,x.end());
        return (x[n] + x[n-1]) / Scalar(2);
    }
    else // otherwise return middle element
        return x[n];
}

// median absolute deviation (MAD)
std::vector<Operon::Scalar> LexicaseSelector::MadFitnesses(Operon::Span<Individual const> pop) const
{
    // Individual.Size() is the number of objectives. will calculate median for each objective
    std::vector<Operon::Scalar> mads(pop[0].Size());

    for (size_t i=0; i<pop[0].Size(); i++) 
    {
        // get fitness of index i in population
        std::vector<Operon::Scalar> x(pop.size()); 
        std::transform(pop.begin(), pop.end(), x.begin(),
            [i](auto &ind){ return ind.Fitness[i]; });

        // median of fitness i
        Operon::Scalar x_median = Median(x);

        Operon::Vector<Operon::Scalar> medians(x.size());

        // calculate MAD for each position
        Operon::Span<Operon::Scalar> dev = { medians.data(), x.size() };
        for (size_t i = 0; i < x.size(); ++i)
            dev[i] = fabs(x[i] - x_median);

        // build the array
        mads.push_back( Median(dev) );
    }

    // return vector of MADs, one for each objective
    return mads;    
}
}  // namespace Operon
