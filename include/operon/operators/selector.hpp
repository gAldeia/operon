// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_SELECTOR_HPP
#define OPERON_SELECTOR_HPP

#include "operon/core/individual.hpp"
#include "operon/core/operator.hpp"
#include "operon/operators/evaluator.hpp"

namespace Operon {
// the selector a vector of individuals and returns the index of a selected individual per each call of operator()
// this operator is meant to be a lightweight object that is initialized with a population and some other parameters on-the-fly
class SelectorBase : public OperatorBase<size_t> {
public:
    using SelectableType = Individual;

    SelectorBase()
        : comp_(nullptr)
    {
    }

    explicit SelectorBase(ComparisonCallback&& cb)
        : comp_(std::move(cb))
    {
    }

    explicit SelectorBase(ComparisonCallback cb)
        : comp_(std::move(cb))
    {
    }

    virtual void Prepare(Operon::Span<Individual const> pop) const
    {
        this->population_ = Operon::Span<const Individual>(pop);
    };

    auto Population() const -> Operon::Span<Individual const> { return population_; }

    [[nodiscard]] inline auto Compare(Individual const& lhs, Individual const& rhs) const -> bool
    {
        return comp_(lhs, rhs);
    }

private:
    mutable Operon::Span<const Individual> population_;
    ComparisonCallback comp_;
};


class OPERON_EXPORT TournamentSelector : public SelectorBase {
public:
    explicit TournamentSelector(ComparisonCallback&& cb) 
        : SelectorBase(cb)
        , tournamentSize_(DefaultTournamentSize)
    { } 
    explicit TournamentSelector(ComparisonCallback const& cb) 
        : SelectorBase(cb)
        , tournamentSize_(DefaultTournamentSize)
    { } 

    auto operator()(Operon::RandomGenerator& random) const -> size_t override;
    void SetTournamentSize(size_t size) { tournamentSize_ = size; }
    auto GetTournamentSize() const -> size_t { return tournamentSize_; }

    static constexpr size_t DefaultTournamentSize = 5;

private:
    size_t tournamentSize_;
};

class OPERON_EXPORT RankTournamentSelector : public SelectorBase {
public:
    explicit RankTournamentSelector(ComparisonCallback&& cb) : SelectorBase(cb){ } 
    explicit RankTournamentSelector(ComparisonCallback const& cb) : SelectorBase(cb){ } 

    auto operator()(Operon::RandomGenerator& random) const -> size_t override;

    void Prepare(Operon::Span<Individual const> pop) const override;

    void SetTournamentSize(size_t size) { tournamentSize_ = size; }
    auto GetTournamentSize() const -> size_t { return tournamentSize_; }

    static constexpr size_t DefaultTournamentSize = 5;

private:
    size_t tournamentSize_{};
    mutable std::vector<size_t> indices_;
};

class OPERON_EXPORT ProportionalSelector : public SelectorBase {
public:
    explicit ProportionalSelector(ComparisonCallback&& cb) : SelectorBase(cb) { } 
    explicit ProportionalSelector(ComparisonCallback const& cb) : SelectorBase(cb) { } 

    auto operator()(Operon::RandomGenerator& random) const -> size_t override;
    
    void Prepare(Operon::Span<Individual const> pop) const override;

    void SetObjIndex(size_t objIndex) { idx_ = objIndex; }
    auto GetObjIndex() const -> size_t { return idx_; }

private:
    void Prepare() const; 

    // discrete CDF of the population fitness values
    mutable std::vector<std::pair<Operon::Scalar, size_t>> fitness_;
    size_t idx_ = 0;
};

class OPERON_EXPORT LexicaseSelector : public SelectorBase {
public:
    explicit LexicaseSelector(ComparisonCallback&& cb, EvaluatorBase& eval)
        : SelectorBase(cb) 
        , evaluator_(eval)
        , range_(eval.GetProblem().TrainingRange())
    {        
    }
    explicit LexicaseSelector(ComparisonCallback const& cb, EvaluatorBase& eval)
        : SelectorBase(cb)
        , evaluator_(eval)
        , range_(eval.GetProblem().TrainingRange())
    {        
    } 

    void Prepare(Operon::Span<Individual const> pop) const override;

    auto operator()(Operon::RandomGenerator& random) const -> size_t override;

    void SetEvaluator(EvaluatorBase& eval) { evaluator_ = eval; }
    auto GetEvaluator() const -> Operon::EvaluatorBase& { return evaluator_.get(); }

private:
    // functions to calculate epsilon
    Operon::Scalar Median(const Operon::Span<Operon::Scalar>& v) const;
    std::vector<Operon::Scalar> MadFitnesses(Operon::Span<Individual const> pop) const;

    // functions to find the pareto front based on epsilon-domination
    bool IsNonDominated(Individual const& lhs, Individual const& rhs, std::vector<Operon::Scalar> eps) const;
    std::vector<std::vector<size_t>> EpsilonDominatedSorter(Operon::Span<Operon::Individual const> pop, std::vector<Operon::Scalar> eps) const;

    // training data range
    Operon::Range range_;

    mutable std::reference_wrapper<Operon::EvaluatorBase> evaluator_;

    // index vectors
    mutable std::vector<size_t> pool_;
    mutable std::vector<size_t> testCases_;
};

class OPERON_EXPORT RandomSelector : public SelectorBase {
public:
    auto operator()(Operon::RandomGenerator& random) const -> size_t override
    {
        return std::uniform_int_distribution<size_t>(0, Population().size() - 1)(random);
    }
};

} //namespace Operon

#endif
