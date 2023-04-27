#if !defined(PLANNERS_APPLICABLE_ACTIONS_NAIVE_GENERATOR_HPP_)
#define PLANNERS_APPLICABLE_ACTIONS_NAIVE_GENERATOR_HPP_

#include "../../formalism/action.hpp"
#include "../../formalism/action_schema.hpp"
#include "../../formalism/problem.hpp"
#include "../../formalism/state.hpp"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace planners
{
    class ApplicableActionsNaiveGenerator
    {
      private:
        using ParameterAssignmentsMap = std::unordered_map<formalism::Parameter, std::unordered_set<formalism::Object>>;

        formalism::ActionSchema action_schema;
        ParameterAssignmentsMap objects_by_parameter_type;

        std::vector<formalism::ParameterAssignment> get_assignments(ParameterAssignmentsMap::const_iterator begin,
                                                                    ParameterAssignmentsMap::const_iterator end) const;

      public:
        ApplicableActionsNaiveGenerator(const formalism::ActionSchema& action_schema, const formalism::ProblemDescription& problem);

        std::vector<formalism::Action> get_applicable_actions(const formalism::State& state) const;
    };
}  // namespace planners

#endif  // PLANNERS_APPLICABLE_ACTIONS_NAIVE_GENERATOR_HPP_
