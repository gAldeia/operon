// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <numeric>
#include "operon/operators/selector.hpp"
#include "operon/core/dataset.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
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

        // TODO: stop using the elite here?
        // Finds the `elite` fitness in pop based on case (x_i, y_i)_{i=caseIdx}, based on comp_
        auto elite = *std::min_element(pop.begin(), pop.end(),
            [this](Individual a, Individual b){ return Compare(a, b); }); 
        
        // estimation of variability of pool using the Median Absolute Deviation (MAD)
        std::vector<Operon::Scalar> MAD = MadFitnesses(pop);

        // TODO: Use NDS here based on my isnondominated (maybe create the epsilon_dominance_sort)?
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
    std::vector<Operon::Scalar> mads(pop.front().Size());

    for (size_t i=0; i<pop.front().Size(); i++) 
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

using Vec = Eigen::Matrix<int64_t, -1, 1, Eigen::ColMajor>;
using Mat = Eigen::Matrix<int64_t, -1, -1, Eigen::ColMajor>;

inline auto ComputeComparisonMatrix(Operon::Span<Operon::Individual const> pop, Mat const& idx, Eigen::Index colIdx) noexcept
{
    auto const n = static_cast<Eigen::Index>(pop.size());
    Mat c = Mat::Zero(n, n);
    Mat::ConstColXpr b = idx.col(colIdx);
    c.row(b(0)).fill(1); // NOLINT
    for (auto i = 1; i < n; ++i) {
        if (pop[b(i)][colIdx] == pop[b(i-1)][colIdx]) {
            c.row(b(i)) = c.row(b(i-1));
        } else {
            for (auto j = i; j < n; ++j) {
                c(b(i), b(j)) = 1;
            }
        }
    }
    return c;
}

inline auto ComparisonMatrixSum(Operon::Span<Operon::Individual const> pop, Mat const& idx) noexcept {
    Mat d = ComputeComparisonMatrix(pop, idx, 0);
    for (int i = 1; i < idx.cols(); ++i) {
        d.noalias() += ComputeComparisonMatrix(pop, idx, i);
    }
    return d;
}

inline auto ComputeDegreeMatrix(Operon::Span<Operon::Individual const> pop, Mat const& idx) noexcept
{
    auto const n = static_cast<Eigen::Index>(pop.size());
    auto const m = static_cast<Eigen::Index>(pop.front().Fitness.size());
    Mat d = ComparisonMatrixSum(pop, idx);
    for (auto i = 0; i < n; ++i) {
        for (auto j = i; j < n; ++j) {
            if (d(i, j) == m && d(j, i) == m) {
                d(i, j) = d(j, i) = 0;
            }
        }
    }
    return d;
}

std::vector<std::vector<size_t>> 
LexicaseSelector::EpsilonDominatedSorter(Operon::Span<Operon::Individual const> pop, std::vector<Operon::Scalar> eps) const
{

    auto const n = static_cast<Eigen::Index>(pop.size());
    auto const m = static_cast<Eigen::Index>(pop.front().Fitness.size());

    Operon::Less cmp;
    Mat idx = Vec::LinSpaced(n, 0, n-1).replicate(1, m);
    for (auto i = 0; i < m; ++i) {
        auto *data = idx.col(i).data();
        std::sort(data, data + n, [&](auto a, auto b) { return cmp(pop[a][i], pop[b][i], eps[i]); });
    }
    Mat d = ComputeDegreeMatrix(pop, idx);

    auto count = 0L; // number of assigned solutions
    std::vector<std::vector<size_t>> fronts;
    std::vector<size_t> tmp(n);
    std::iota(tmp.begin(), tmp.end(), 0UL);

    std::vector<size_t> remaining;
    while (count < n) {
        std::vector<size_t> front;
        for (auto i : tmp) { // TODO: i think i need to use my isNonDominated here
            if (std::all_of(tmp.begin(), tmp.end(), [&](auto j) { return d(j, i) < m; })) {
                front.push_back(i);
            } else {
                remaining.push_back(i);
            }
        }
        tmp.swap(remaining);
        remaining.clear();
        count += static_cast<int64_t>(front.size());
        fronts.push_back(front);
    }
    return fronts;
}
}  // namespace Operon
