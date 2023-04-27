/*
 * Copyright (C) 2023 Simon Stahlberg
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */


#include "../algorithms/weisfeiler_leman.hpp"
#include "../formalism/help_functions.hpp"
#include "generators/successor_generator_factory.hpp"
#include "state_space.hpp"
#include "system.hpp"

#include <chrono>
#include <deque>

#if defined(__linux__)
#include <malloc.h>
#endif

namespace std
{
    template<>
    struct hash<pair<uint64_t, uint64_t>>
    {
        size_t operator()(const pair<uint64_t, uint64_t>& x) const { return (size_t) x.first; }
    };
}  // namespace std

namespace planners
{
    struct StateInfo
    {
        int32_t distance_from_initial_state;
        int32_t distance_to_goal_state;

        StateInfo(int32_t distance_from_initial_state, int32_t distance_to_goal_state) :
            distance_from_initial_state(distance_from_initial_state),
            distance_to_goal_state(distance_to_goal_state)
        {
        }
    };

    StateSpaceImpl::StateSpaceImpl(const formalism::ProblemDescription& problem) :
        states_(),
        goal_states_(),
        forward_transitions_(),
        backward_transitions_(),
        state_indices_(),
        domain(problem->domain),
        problem(problem)
    {
    }

    StateSpaceImpl::~StateSpaceImpl()
    {
        states_.clear();
        goal_states_.clear();
        state_infos_.clear();
        forward_transitions_.clear();
        backward_transitions_.clear();
        state_indices_.clear();
    }

    bool StateSpaceImpl::add_or_get_state(const formalism::State& state, uint64_t& out_index)
    {
        const auto index_handler = state_indices_.find(state);

        if (index_handler != state_indices_.end())
        {
            out_index = index_handler->second;
            return false;
        }
        else
        {
            out_index = states_.size();
            states_.push_back(state);
            state_infos_.push_back(StateInfo(-1, -1));
            forward_transitions_.push_back(std::vector<formalism::Transition>());
            backward_transitions_.push_back(std::vector<formalism::Transition>());
            state_indices_.insert(std::make_pair(state, out_index));

            if (literals_hold(problem->goal, state))
            {
                goal_states_.push_back(state);
            }

            return true;
        }
    }

    void StateSpaceImpl::add_goal_state(const formalism::State& state)
    {
        if (std::find(goal_states_.begin(), goal_states_.end(), state) == goal_states_.end())
        {
            goal_states_.push_back(state);
        }
    }

    void StateSpaceImpl::add_transition(uint64_t from_state_index,
                                        uint64_t to_state_index,
                                        const formalism::Action& action,
                                        uint64_t& out_from_forward_index,
                                        uint64_t& out_to_backward_index)
    {
        out_from_forward_index = forward_transitions_[from_state_index].size();
        out_to_backward_index = backward_transitions_[to_state_index].size();

        const auto transition = create_transition(get_state(from_state_index), action, get_state(to_state_index));
        forward_transitions_[from_state_index].push_back(transition);
        backward_transitions_[to_state_index].push_back(transition);
    }

    const formalism::Transition& StateSpaceImpl::get_forward_transition(uint64_t state_index, uint64_t transition_index) const
    {
        return forward_transitions_[state_index][transition_index];
    }

    const formalism::Transition& StateSpaceImpl::get_backward_transition(uint64_t state_index, uint64_t transition_index) const
    {
        return backward_transitions_[state_index][transition_index];
    }

    const formalism::TransitionList& StateSpaceImpl::get_forward_transitions(uint64_t state_index) const { return forward_transitions_[state_index]; }

    const formalism::TransitionList& StateSpaceImpl::get_backward_transitions(uint64_t state_index) const { return backward_transitions_[state_index]; }

    formalism::State StateSpaceImpl::get_state(uint64_t state_index) const { return states_[state_index]; }

    int32_t StateSpaceImpl::get_distance_to_goal_state(uint64_t state_index) const { return state_infos_[state_index].distance_to_goal_state; }

    int32_t StateSpaceImpl::get_longest_distance_to_goal_state() const
    {
        int32_t longest_distance_to_goal_state = 0;

        for (std::size_t index = 0; index < state_infos_.size(); ++index)
        {
            longest_distance_to_goal_state = std::max(longest_distance_to_goal_state, state_infos_[index].distance_to_goal_state);
        }

        return longest_distance_to_goal_state;
    }

    std::vector<uint32_t> StateSpaceImpl::get_distance_to_goal_state_histogram() const
    {
        std::vector<uint32_t> histogram;
        histogram.resize(get_longest_distance_to_goal_state() + 1);

        for (std::size_t index = 0; index < state_infos_.size(); ++index)
        {
            const auto distance = state_infos_[index].distance_to_goal_state;

            if (distance >= 0)
            {
                ++histogram[distance];
            }
        }

        return histogram;
    }

    std::vector<double> StateSpaceImpl::get_distance_to_goal_state_weights() const
    {
        const auto histogram = get_distance_to_goal_state_histogram();
        const auto largest_class = (double) *std::max_element(histogram.begin(), histogram.end());

        std::vector<double> weights;
        weights.resize(histogram.size(), 1.0);

        for (std::size_t index = 0; index < histogram.size(); ++index)
        {
            if (histogram[index] > 0)
            {
                weights[index] = largest_class / (double) histogram[index];
            }
        }

        return weights;
    }

    int32_t StateSpaceImpl::get_distance_from_initial_state(uint64_t state_index) const { return state_infos_[state_index].distance_from_initial_state; }

    const std::vector<formalism::Transition>& StateSpaceImpl::get_forward_transitions(const formalism::State& state) const
    {
        const auto index = get_state_index(state);
        return forward_transitions_[index];
    }

    const std::vector<formalism::Transition>& StateSpaceImpl::get_backward_transitions(const formalism::State& state) const
    {
        const auto index = get_state_index(state);
        return backward_transitions_[index];
    }

    const std::vector<formalism::State>& StateSpaceImpl::get_states() const { return states_; }

    formalism::State StateSpaceImpl::get_initial_state() const { return problem->initial; }

    uint64_t StateSpaceImpl::get_unique_index_of_state(const formalism::State& state) const { return get_state_index(state); }

    uint64_t StateSpaceImpl::get_state_index(const formalism::State& state) const
    {
        const auto index_handler = state_indices_.find(state);

        if (index_handler == state_indices_.end())
        {
            throw std::invalid_argument("state");
        }
        else
        {
            return index_handler->second;
        }
    }

    bool StateSpaceImpl::is_dead_end_state(const formalism::State& state) const
    {
        const auto index = get_state_index(state);
        const auto& info = state_infos_[index];
        return info.distance_to_goal_state < 0;
    }

    bool StateSpaceImpl::is_goal_state(const formalism::State& state) const
    {
        const auto index = get_state_index(state);
        const auto& info = state_infos_[index];
        return info.distance_to_goal_state == 0;
    }

    int32_t StateSpaceImpl::get_distance_to_goal_state(const formalism::State& state) const
    {
        const auto index = get_state_index(state);
        return get_distance_to_goal_state(index);
    }

    int32_t StateSpaceImpl::get_distance_from_initial_state(const formalism::State& state) const
    {
        const auto index = get_state_index(state);
        return get_distance_from_initial_state(index);
    }

    void StateSpaceImpl::set_distance_from_initial_state(uint64_t state_index, int32_t value) { state_infos_[state_index].distance_from_initial_state = value; }

    void StateSpaceImpl::set_distance_to_goal_state(uint64_t state_index, int32_t value) { state_infos_[state_index].distance_to_goal_state = value; }

    std::vector<formalism::State> StateSpaceImpl::get_goal_states() const { return goal_states_; }

    uint64_t StateSpaceImpl::num_states() const { return (uint64_t) states_.size(); }

    uint64_t StateSpaceImpl::num_transitions() const
    {
        uint64_t num_transitions = 0;

        for (std::size_t i = 0; i < forward_transitions_.size(); ++i)
        {
            num_transitions += forward_transitions_[i].size();
        }

        return num_transitions;
    }

    uint64_t StateSpaceImpl::num_goal_states() const { return (uint64_t) goal_states_.size(); }

    uint64_t StateSpaceImpl::num_dead_end_states() const
    {
        uint64_t num_dead_ends = 0;

        for (const auto& info : state_infos_)
        {
            if (info.distance_to_goal_state < 0)
            {
                ++num_dead_ends;
            }
        }

        return num_dead_ends;
    }

    StateSpace create_state_space(const formalism::ProblemDescription& problem,
                                  uint32_t max_states,
                                  bool prune_with_weisfeiler_leman,
                                  int32_t timeout_s,
                                  int32_t max_memory_mb)
    {
#if defined(__linux__)
        // If possible, release allocated but unused memory back to the OS so that we can keep track of memory usage accurately.
        // However, there is no platform-independent way of doing this.
        // TODO: Implement a custom allocator that keep tracks of used and unused allocated memory to abort create_state_space more intelligently.
        malloc_trim(0);
#endif

        const auto initial_memory_usage_mb = static_cast<int32_t>((resources::get_memory_usage() / 1024) / 1024);
        auto state_space = new StateSpaceImpl(problem);
        const auto successor_generator = planners::create_sucessor_generator(problem, planners::SuccessorGeneratorType::AUTOMATIC);

        std::vector<uint64_t> goal_indices;
        std::vector<bool> is_expanded;
        std::deque<uint64_t> queue;

        algorithms::WeisfeilerLeman wl;
        std::unordered_map<std::pair<uint64_t, uint64_t>, uint64_t> wl_mapping;

        {
            uint64_t initial_index;

            if (state_space->add_or_get_state(problem->initial, initial_index))
            {
                state_space->set_distance_from_initial_state(initial_index, 0);
                is_expanded.push_back(false);

                if (prune_with_weisfeiler_leman)
                {
                    const auto initial_color = wl.compute_state_color(problem, problem->initial);
                    wl_mapping.insert(std::make_pair(initial_color, initial_index));
                }
            }

            queue.push_back(initial_index);
        }

        const auto goal = problem->goal;
        auto start_time = std::chrono::high_resolution_clock::now();

        while ((queue.size() > 0) && (state_space->num_states() < max_states))
        {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            const auto current_memory_usage_mb = static_cast<int32_t>((resources::get_memory_usage() / 1024) / 1024);

            if ((timeout_s > 0) && (elapsed_time >= timeout_s))
            {
                std::cerr << "Timeout reached: " << problem->name;
                std::cerr << ", initial memory: " << initial_memory_usage_mb << " MB";
                std::cerr << ", current memory: " << current_memory_usage_mb << " MB" << std::endl;
                delete state_space;
                return nullptr;
            }

            if ((max_memory_mb > 0) && (current_memory_usage_mb > max_memory_mb))
            {
                std::cerr << "Exceeded memory limit: " << problem->name;
                std::cerr << ", initial memory: " << initial_memory_usage_mb << " MB";
                std::cerr << ", current memory: " << current_memory_usage_mb << " MB" << std::endl;
                delete state_space;
                return nullptr;
            }

            const auto state_index = queue.front();
            queue.pop_front();

            if (is_expanded[state_index])
            {
                continue;
            }

            const auto state = state_space->get_state(state_index);
            const auto distance_from_initial_state = state_space->get_distance_from_initial_state(state_index);
            const auto is_goal_state = formalism::literals_hold(goal, state);

            if (is_goal_state)
            {
                goal_indices.push_back(state_index);
                state_space->add_goal_state(state);
            }

            is_expanded[state_index] = true;
            const auto ground_actions = successor_generator->get_applicable_actions(state);
            std::vector<uint64_t> successor_indices;

            for (const auto& ground_action : ground_actions)
            {
                const auto successor_state = formalism::apply(ground_action, state);
                uint64_t successor_state_index;
                bool successor_is_new_state;

                if (prune_with_weisfeiler_leman)
                {
                    const auto successor_color = wl.compute_state_color(problem, successor_state);

                    if (wl_mapping.count(successor_color) == 0)
                    {
                        // We've not seen an isomorphic state before, proceed as usual.
                        successor_is_new_state = state_space->add_or_get_state(successor_state, successor_state_index);
                        wl_mapping.insert(std::make_pair(successor_color, successor_state_index));
                    }
                    else
                    {
                        // We've seen an isomorphic state before, replace the successor with it.
                        successor_state_index = wl_mapping.at(successor_color);
                        successor_is_new_state = false;
                    }
                }
                else
                {
                    successor_is_new_state = state_space->add_or_get_state(successor_state, successor_state_index);
                }

                // Do not add the transition to successor if it already exists when pruning with Weisfeiler-Leman.
                if (prune_with_weisfeiler_leman)
                {
                    const auto has_transition = std::count(successor_indices.begin(), successor_indices.end(), successor_state_index) > 0;

                    if (has_transition)
                    {
                        continue;
                    }
                    else
                    {
                        successor_indices.push_back(successor_state_index);
                    }
                }

                uint64_t from_transition_index, to_transition_index;
                state_space->add_transition(state_index, successor_state_index, ground_action, from_transition_index, to_transition_index);

                if (successor_is_new_state)
                {
                    state_space->set_distance_from_initial_state(successor_state_index, distance_from_initial_state + 1);
                    is_expanded.push_back(false);
                    queue.push_back(successor_state_index);
                }
            }
        }

        // Check if every state was expanded (i.e., if max_states stopped the search).

        if (queue.size() > 0)
        {
            delete state_space;
            return nullptr;
        }

        queue.insert(queue.end(), goal_indices.begin(), goal_indices.end());

        for (const auto& goal_state_index : goal_indices)
        {
            state_space->set_distance_to_goal_state(goal_state_index, 0);
        }

        std::fill(is_expanded.begin(), is_expanded.end(), false);

        while (queue.size() > 0)
        {
            const auto state_index = queue.front();
            queue.pop_front();

            if (is_expanded[state_index])
            {
                continue;
            }

            is_expanded[state_index] = true;

            const auto distance_to_goal_state = state_space->get_distance_to_goal_state(state_index);
            const auto& backward_transitions = state_space->get_backward_transitions(state_index);

            for (const auto& backward_transition : backward_transitions)
            {
                const auto predecessor_state_index = state_space->get_state_index(backward_transition->source_state);
                const auto predecessor_is_new_state = state_space->get_distance_to_goal_state(predecessor_state_index) < 0;

                if (predecessor_is_new_state)
                {
                    state_space->set_distance_to_goal_state(predecessor_state_index, distance_to_goal_state + 1);
                    queue.push_back(predecessor_state_index);
                }
            }
        }

        return StateSpace(state_space);
    }

    StateSpace prune_state_space_with_weisfeiler_leman(const StateSpace& state_space)
    {
        algorithms::WeisfeilerLeman wl;
        const auto problem = state_space->problem;
        std::unordered_map<std::pair<uint64_t, uint64_t>, formalism::State> state_classes;
        std::vector<formalism::State> distinct_states;

        {  // Ensure that the initial state of the problem is the "representative" of its class
            const auto initial_state = state_space->get_initial_state();
            const auto initial_color = wl.compute_state_color(problem, initial_state);
            state_classes.insert(std::make_pair(initial_color, initial_state));
            distinct_states.push_back(initial_state);
        }

        for (const auto& state : state_space->get_states())
        {
            const auto color = wl.compute_state_color(problem, state);
            const auto state_class_handler = state_classes.find(color);

            if (state_class_handler != state_classes.end())
            {
                const auto class_state = state_class_handler->second;
                const auto class_distance_init = state_space->get_distance_from_initial_state(class_state);
                const auto state_distance_init = state_space->get_distance_from_initial_state(state);
                const auto class_distance_goal = state_space->get_distance_to_goal_state(class_state);
                const auto state_distance_goal = state_space->get_distance_to_goal_state(state);

                if ((class_distance_init != state_distance_init) || (class_distance_goal != state_distance_goal))
                {
                    // There are two states with a different distance to the closest goal state.
                    // This means 1-WL pruning cannot be performed safely.
                    // Return a nullptr to signal this.
                    return nullptr;
                }
            }
            else
            {
                state_classes.insert(std::make_pair(color, state));
                distinct_states.push_back(state);
            }
        }

        std::shared_ptr<StateSpaceImpl> pruned_state_space(new StateSpaceImpl(problem));

        for (const auto& state : distinct_states)
        {
            const auto from_initial = state_space->get_distance_from_initial_state(state);
            const auto to_goal = state_space->get_distance_to_goal_state(state);

            uint64_t state_index;
            pruned_state_space->add_or_get_state(state, state_index);
            pruned_state_space->set_distance_from_initial_state(state_index, from_initial);
            pruned_state_space->set_distance_to_goal_state(state_index, to_goal);

            if (to_goal == 0)
            {
                pruned_state_space->add_goal_state(state);
            }
        }

        for (const auto& from_state : distinct_states)
        {
            const auto from_index = pruned_state_space->get_state_index(from_state);
            std::unordered_set<std::pair<uint64_t, uint64_t>> to_colors;

            for (const auto& transition : state_space->get_forward_transitions(from_state))
            {
                const auto to_color = wl.compute_state_color(problem, transition->target_state);

                if (to_colors.count(to_color) == 0)
                {
                    to_colors.insert(to_color);
                    const auto state_class_handler = state_classes.find(to_color);
                    const auto to_state = state_class_handler->second;
                    const auto to_index = pruned_state_space->get_state_index(to_state);

                    uint64_t from_forward_index, to_backward_index;
                    pruned_state_space->add_transition(from_index, to_index, transition->action, from_forward_index, to_backward_index);
                }
            }
        }

        return pruned_state_space;
    }

    std::ostream& operator<<(std::ostream& os, const planners::StateSpace& state_space)
    {
        os << "# States: " << state_space->num_states() << "; # Transitions: " << state_space->num_transitions();
        return os;
    }

    std::ostream& operator<<(std::ostream& os, const planners::StateSpaceList& state_spaces)
    {
        print_vector(os, state_spaces);
        return os;
    }
}  // namespace planners
