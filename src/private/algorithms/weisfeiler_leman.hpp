#if !defined(ALGORITHMS_WEISFEILER_LEMAN_HPP_)
#define ALGORITHMS_WEISFEILER_LEMAN_HPP_

#include "../formalism/problem.hpp"
#include "../formalism/state.hpp"

#include <stdint.h>
#include <tuple>
#include <vector>

namespace algorithms
{
    class WLGraph
    {
      public:
        const std::vector<int32_t> vertices;
        const std::vector<std::tuple<int32_t, int32_t, int32_t>> edges;
        WLGraph(const std::vector<int32_t>& vertices, const std::vector<std::tuple<int32_t, int32_t, int32_t>>& edges) : vertices(vertices), edges(edges) {}
    };

    class WeisfeilerLeman
    {
      private:
        const int32_t k_;
        std::map<std::pair<int32_t, std::vector<std::pair<int32_t, int32_t>>>, int32_t> hash_;
        std::map<std::string, int32_t> predicate_ids_;
        std::map<std::vector<int32_t>, int32_t> label_ids_;

        void add_atom(const formalism::Atom& atom,
                      const std::string& predicate_suffix,
                      std::map<std::string, int32_t>& object_ids,
                      std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>>& edge_labels,
                      int32_t& variable_counter);

      public:
        WeisfeilerLeman();

        WeisfeilerLeman(int32_t k);

        std::vector<int32_t> compute_color_histogram(const WLGraph& graph);

        std::pair<uint64_t, uint64_t> hash_color_histogram(const std::vector<int32_t>& histogram);

        WLGraph to_wl_graph(const formalism::ProblemDescription& problem, const formalism::State& state);

        std::pair<uint64_t, uint64_t> compute_state_color(const formalism::ProblemDescription& problem, const formalism::State& state);
    };

}  // namespace algorithms

#endif  // ALGORITHMS_WEISFEILER_LEMAN_HPP_
