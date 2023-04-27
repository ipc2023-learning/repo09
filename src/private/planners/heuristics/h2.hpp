#if !defined(PLANNERS_HEURISTIC_H2_HPP_)
#define PLANNERS_HEURISTIC_H2_HPP_

#include "../../formalism/atom.hpp"
#include "../../formalism/problem.hpp"
#include "../../formalism/state.hpp"
#include "heuristic_base.hpp"

#include <limits>
#include <unordered_map>
#include <vector>

namespace planners
{
    struct H2Action
    {
        std::vector<int32_t> preconditions;
        std::vector<int32_t> adds;
        std::vector<int32_t> deletes;
        std::vector<int32_t> deletesComplement;
        int32_t cost;

        H2Action() : preconditions(), adds(), deletesComplement(), cost(1) {}
    };

    class H2Heuristic : public HeuristicBase
    {
      private:
        static constexpr int32_t INTERNAL_DEAD_END = std::numeric_limits<int32_t>::max();

        std::vector<std::vector<bool>> partitions_;
        std::vector<H2Action> actions_;
        std::vector<int32_t> goal_;
        mutable std::map<formalism::Atom, int32_t> atom_ids_;
        mutable std::vector<int32_t> ht1_;
        mutable std::vector<std::vector<int32_t>> ht2_;

        int32_t get_id(const formalism::Atom& atom) const;

        int32_t evaluate(const std::vector<int32_t>& s) const;
        int32_t evaluate(const std::vector<int32_t>& s, int32_t x) const;
        void update(const std::size_t val, const int32_t h, bool& changed) const;
        void update(const std::size_t val1, const std::size_t val2, const int32_t h, bool& changed) const;
        void fill_tables(const std::vector<bool>& partition, const formalism::State& state) const;

      public:
        H2Heuristic(const formalism::ProblemDescription& problem);

        double get_cost(const formalism::State& state) const override;
    };
}  // namespace planners

#endif  // PLANNERS_HEURISTIC_H2_HPP_
