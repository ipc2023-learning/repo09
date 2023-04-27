#if !defined(PLANNERS_STATE_SPACE_HPP_)
#define PLANNERS_STATE_SPACE_HPP_

#include "../formalism/declarations.hpp"
#include "../formalism/domain.hpp"
#include "../formalism/problem.hpp"
#include "../formalism/state.hpp"
#include "../formalism/transition.hpp"

#include <limits>
#include <unordered_map>
#include <vector>

namespace planners
{
    class StateSpaceImpl;
    using StateSpace = std::shared_ptr<StateSpaceImpl>;
    using StateSpaceList = std::vector<StateSpace>;
    using StateSpaceSample = std::pair<formalism::State, StateSpace>;
    using StateSpaceSampleList = std::vector<StateSpaceSample>;

    struct StateInfo;

    class StateSpaceImpl
    {
      private:
        std::vector<formalism::State> states_;
        std::vector<formalism::State> goal_states_;
        std::vector<StateInfo> state_infos_;
        std::vector<std::vector<formalism::Transition>> forward_transitions_;
        std::vector<std::vector<formalism::Transition>> backward_transitions_;
        std::unordered_map<formalism::State, uint64_t> state_indices_;

        // Since we return references of internal vectors, ensure that only create_statespaces can create this object.
        StateSpaceImpl(const formalism::ProblemDescription& problem);

        bool add_or_get_state(const formalism::State& state, uint64_t& out_index);

        void add_goal_state(const formalism::State& state);

        void add_transition(uint64_t from_state_index,
                            uint64_t to_state_index,
                            const formalism::Action& action,
                            uint64_t& out_from_forward_index,
                            uint64_t& out_to_backward_index);

        const formalism::Transition& get_forward_transition(uint64_t state_index, uint64_t transition_index) const;

        const formalism::Transition& get_backward_transition(uint64_t state_index, uint64_t transition_index) const;

        const formalism::TransitionList& get_forward_transitions(uint64_t state_index) const;

        const formalism::TransitionList& get_backward_transitions(uint64_t state_index) const;

        formalism::State get_state(uint64_t state_index) const;

        uint64_t get_state_index(const formalism::State& state) const;

        int32_t get_distance_to_goal_state(uint64_t state_index) const;

        int32_t get_distance_from_initial_state(uint64_t state_index) const;

        void set_distance_from_initial_state(uint64_t state_index, int32_t value);

        void set_distance_to_goal_state(uint64_t state_index, int32_t value);

      public:
        formalism::DomainDescription domain;
        formalism::ProblemDescription problem;

        virtual ~StateSpaceImpl();

        const std::vector<formalism::Transition>& get_forward_transitions(const formalism::State& state) const;

        const std::vector<formalism::Transition>& get_backward_transitions(const formalism::State& state) const;

        const std::vector<formalism::State>& get_states() const;

        formalism::State get_initial_state() const;

        uint64_t get_unique_index_of_state(const formalism::State& state) const;

        bool is_dead_end_state(const formalism::State& state) const;

        bool is_goal_state(const formalism::State& state) const;

        int32_t get_distance_to_goal_state(const formalism::State& state) const;

        int32_t get_longest_distance_to_goal_state() const;

        std::vector<uint32_t> get_distance_to_goal_state_histogram() const;

        std::vector<double> get_distance_to_goal_state_weights() const;

        int32_t get_distance_from_initial_state(const formalism::State& state) const;

        std::vector<formalism::State> get_goal_states() const;

        uint64_t num_states() const;

        uint64_t num_transitions() const;

        uint64_t num_goal_states() const;

        uint64_t num_dead_end_states() const;

        friend StateSpace create_state_space(const formalism::ProblemDescription&, uint32_t, bool, int32_t, int32_t);

        friend StateSpace prune_state_space_with_weisfeiler_leman(const StateSpace&);
    };

    StateSpace create_state_space(const formalism::ProblemDescription& problem,
                                  uint32_t max_states = std::numeric_limits<uint32_t>::max(),
                                  bool prune_with_weisfeiler_leman = false,
                                  int32_t timeout_s = -1,
                                  int32_t max_memory_mb = -1);

    StateSpace prune_state_space_with_weisfeiler_leman(const StateSpace& state_space);

    std::ostream& operator<<(std::ostream& os, const planners::StateSpace& state_space);

    std::ostream& operator<<(std::ostream& os, const planners::StateSpaceList& state_spaces);

}  // namespace planners

#endif  // PLANNERS_STATE_SPACE_HPP_
