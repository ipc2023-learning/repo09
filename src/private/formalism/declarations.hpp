#if !defined(FORMALISM_DECLARATIONS_HPP_)
#define FORMALISM_DECLARATIONS_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace formalism
{
    class ActionImpl;
    using Action = std::shared_ptr<ActionImpl>;
    using ActionList = std::vector<Action>;

    class ActionSchemaImpl;
    using ActionSchema = std::shared_ptr<ActionSchemaImpl>;
    using ActionSchemaList = std::vector<ActionSchema>;

    class AtomImpl;
    using Atom = std::shared_ptr<AtomImpl>;
    using AtomList = std::vector<Atom>;
    using AtomSet = std::unordered_set<formalism::Atom>;

    class DomainImpl;
    using DomainDescription = std::shared_ptr<DomainImpl>;
    using Requirement = std::string;
    using RequirementList = std::vector<Requirement>;

    class LiteralImpl;
    using Literal = std::shared_ptr<LiteralImpl>;
    using LiteralList = std::vector<Literal>;

    class ObjectImpl;
    using Object = std::shared_ptr<ObjectImpl>;
    using ObjectList = std::vector<Object>;
    using Parameter = Object;
    using ParameterList = ObjectList;
    using ParameterAssignment = std::unordered_map<Parameter, Object>;

    class PredicateImpl;
    using Predicate = std::shared_ptr<PredicateImpl>;
    using PredicateList = std::vector<Predicate>;
    using PredicateSet = std::unordered_set<formalism::Predicate>;

    class ProblemImpl;
    using ProblemDescription = std::shared_ptr<ProblemImpl>;
    using ProblemDescriptionList = std::vector<ProblemDescription>;
    using Requirement = std::string;
    using RequirementList = std::vector<Requirement>;

    class StateImpl;
    using State = std::shared_ptr<StateImpl>;
    using StateList = std::vector<State>;
    using StateTransitions = std::tuple<State, StateList, ProblemDescription>;
    using StateTransitionsVector = std::vector<StateTransitions>;
    using StateProblem = std::pair<State, ProblemDescription>;
    using StateProblemList = std::vector<StateProblem>;

    class TransitionImpl;
    using Transition = std::shared_ptr<TransitionImpl>;
    using TransitionList = std::vector<Transition>;

}  // namespace formalism

#endif  // FORMALISM_DECLARATIONS_HPP_
