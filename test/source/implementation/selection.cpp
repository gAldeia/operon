// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research
//
#include <algorithm>
#include <doctest/doctest.h>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/core/pset.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/selector.hpp"
#include "operon/operators/initializer.hpp"

namespace Operon::Test {

TEST_CASE("Selection Distribution")
{
    size_t nTrees = 100;
    size_t maxLength = 100;
    size_t maxDepth = 12;

    Operon::RandomGenerator random(1234);

    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);

    auto dataset = Dataset("../../../data/Poly-10.csv", /*hasHeader=*/true);
    auto inputs = dataset.VariableHashes();

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);
    grammar.SetMaximumArity(Node(NodeType::Add), 2);
    grammar.SetMaximumArity(Node(NodeType::Mul), 2);
    grammar.SetMaximumArity(Node(NodeType::Sub), 2);
    grammar.SetMaximumArity(Node(NodeType::Div), 2);

    grammar.SetFrequency(Node(NodeType::Add), 4);
    grammar.SetFrequency(Node(NodeType::Mul), 1);
    grammar.SetFrequency(Node(NodeType::Sub), 1);
    grammar.SetFrequency(Node(NodeType::Div), 1);

    auto fullRange = Operon::Range{ 0, 2 * dataset.Rows<std::size_t>() / 3 };

    Operon::Problem problem(dataset, fullRange, fullRange);
    problem.ConfigurePrimitiveSet(Operon::PrimitiveSet::Arithmetic);

    auto const& [error, scale] = std::make_tuple(Operon::MAE(), false);
    Operon::Interpreter interpreter;
    Operon::Evaluator evaluator(problem, interpreter, error, scale);

    auto creator = BalancedTreeCreator { grammar, inputs };

    std::vector<size_t> lengths(nTrees);
    std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
    
    using Dist = std::normal_distribution<Operon::Scalar>;
    auto coeffInitializer = Operon::CoefficientInitializer<Dist>();
    dynamic_cast<Operon::NormalCoefficientInitializer*>(&coeffInitializer)->ParameterizeDistribution(Operon::Scalar{0}, Operon::Scalar{1});

    Operon::Scalar* data = nullptr;
    Operon::Span<Operon::Scalar> buf;
    
    Operon::Vector<Individual> individuals; //(nTrees);
    std::transform(lengths.begin(), lengths.end(), std::back_inserter(individuals), [&](size_t len) {
        Individual ind;
        ind.Genotype = creator(random, len, 0, maxDepth);
        coeffInitializer(random, ind.Genotype);

        ind.Fitness = evaluator(random, ind, buf);

        return ind;
    });

    Operon::Span<Individual> pop(individuals);

    // using Ind = Individual<1>;
    auto comp = [](auto const& lhs, auto const& rhs) { return lhs[0] < rhs[0]; };

    ProportionalSelector proportionalSelector(comp);
    proportionalSelector.Prepare(pop);

    TournamentSelector tournamentSelector(comp);
    tournamentSelector.Prepare(pop);

    RankTournamentSelector rankedSelector(comp);
    rankedSelector.Prepare(pop);

    LexicaseSelector lexicaseSelector(comp, evaluator);
    lexicaseSelector.Prepare(pop);

    auto plotHist = [&](SelectorBase& selector)
    {
        std::vector<size_t> hist(pop.size());

        for (size_t i = 0; i < 100 * nTrees; ++i)
        {
            hist[selector(random)]++;
        }
        std::sort(hist.begin(), hist.end(), std::greater<>{});
        for (size_t i = 0; i < nTrees; ++i)
        {
            //auto qty = std::string(hist[i], '*');
            auto qty = hist[i];
            fmt::print("{:>5}\t{}\n", i, qty / 100.0);
        }
    };

    SUBCASE("Proportional")
    {
        plotHist(proportionalSelector);
    }

    SUBCASE("Tournament Size 2")
    {
        plotHist(tournamentSelector);
    }

    SUBCASE("Rank Tournament Size 2")
    {
        plotHist(rankedSelector);
    }
    
    SUBCASE("Tournament Size 3")
    {
        tournamentSelector.SetTournamentSize(3);
        plotHist(tournamentSelector);
    }

    SUBCASE("Rank Tournament Size 3")
    {
        rankedSelector.SetTournamentSize(3);
        plotHist(rankedSelector);
    }
    
    SUBCASE("epsilon-Lexicase")
    {
        plotHist(lexicaseSelector);
    }
}
}
