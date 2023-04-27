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


#include "abstract_syntax_tree.hpp"

#include <boost/fusion/include/at_c.hpp>

namespace parsers
{
    /* ASTNode */

    ASTNode::~ASTNode() {};

    /* CharacterNode */

    CharacterNode::CharacterNode(char character) : character(character) {}

    /* NameNode */

    NameNode::NameNode(char prefix, const std::vector<CharacterNode*>& character_nodes) : prefix(prefix), character_nodes(character_nodes) {}

    NameNode::~NameNode()
    {
        for (auto node : character_nodes)
        {
            if (node)
            {
                delete node;
            }
        }

        character_nodes.clear();
    }

    std::string NameNode::get_name() const
    {
        std::vector<char> characters;
        characters.push_back(prefix);

        for (auto node : character_nodes)
        {
            characters.push_back(node->character);
        }

        return std::string(characters.begin(), characters.end());
    }

    /* VariableNode */

    VariableNode::VariableNode(NameNode* name_node) : name_node(name_node) {}

    VariableNode::~VariableNode()
    {
        if (name_node)
        {
            delete name_node;
            name_node = nullptr;
        }
    }

    std::string VariableNode::get_variable() const { return "?" + name_node->get_name(); }

    /* TypeNode */

    TypeNode::TypeNode(std::string type_name) : name_string(type_name), name_node(nullptr) {}

    TypeNode::TypeNode(NameNode* type_name) : name_string(), name_node(type_name) {}

    TypeNode::~TypeNode()
    {
        if (name_node)
        {
            delete name_node;
            name_node = nullptr;
        }
    }

    std::string TypeNode::get_type() const
    {
        if (name_node)
        {
            return name_node->get_name();
        }
        else
        {
            return name_string;
        }
    }

    /* TermNode */

    TermNode::TermNode(NameNode* node) : name_node(node), variable_node(nullptr) {}

    TermNode::TermNode(VariableNode* node) : name_node(nullptr), variable_node(node) {}

    TermNode::~TermNode()
    {
        if (name_node)
        {
            delete name_node;
            name_node = nullptr;
        }

        if (variable_node)
        {
            delete variable_node;
            variable_node = nullptr;
        }
    }

    formalism::Object TermNode::get_term(const std::map<std::string, formalism::Parameter>& parameters,
                                         const std::map<std::string, formalism::Object>& constants) const
    {
        std::string name;

        if (name_node)
        {
            name = name_node->get_name();
        }
        else if (variable_node)
        {
            name = variable_node->get_variable();
        }
        else
        {
            throw std::runtime_error("internal error: both name_node and variable_node are null");
        }

        // TODO: Check parameters and thne constants
        const auto parameters_handler = parameters.find(name);
        if (parameters_handler != parameters.end())
        {
            return parameters_handler->second;
        }

        const auto constants_handler = constants.find(name);
        if (constants_handler != constants.end())
        {
            return constants_handler->second;
        }

        throw std::invalid_argument("the argument \"" + name + "\" is undefined");
    }

    /* AtomNode */

    AtomNode::AtomNode(NameNode* name, std::vector<TermNode*>& arguments) : name(name), arguments(arguments) {}

    AtomNode::~AtomNode()
    {
        if (name)
        {
            delete name;
            name = nullptr;
        }

        for (auto node : arguments)
        {
            delete node;
        }
        arguments.clear();
    }

    formalism::Atom AtomNode::get_atom(const std::map<std::string, formalism::Parameter>& parameters,
                                       const std::map<std::string, formalism::Object>& constants,
                                       const std::map<std::string, formalism::Predicate>& predicates) const
    {
        const auto atom_name = name->get_name();
        const auto predicate_handler = predicates.find(atom_name);

        if (predicate_handler != predicates.end())
        {
            const auto predicate = predicate_handler->second;
            formalism::ObjectList atom_arguments;

            for (const auto node : arguments)
            {
                atom_arguments.push_back(node->get_term(parameters, constants));
            }

            return formalism::create_atom(predicate, atom_arguments);
        }
        else
        {
            throw std::invalid_argument("the predicate of the atom \"" + atom_name + "\" is undefined");
        }
    }

    /* TypedNameListNode */

    TypedNameListNode::TypedNameListNode(std::vector<NameNode*>& untyped_names) : untyped_names(untyped_names), typed_names(), type(nullptr), recursion(nullptr)
    {
    }

    TypedNameListNode::TypedNameListNode(std::vector<NameNode*>& typed_names, TypeNode* type, TypedNameListNode* recursion) :
        untyped_names(),
        typed_names(typed_names),
        type(type),
        recursion(recursion)
    {
    }

    TypedNameListNode::~TypedNameListNode()
    {
        for (auto name : untyped_names)
        {
            delete name;
        }
        untyped_names.clear();

        for (auto name : typed_names)
        {
            delete name;
        }
        typed_names.clear();

        if (type)
        {
            delete type;
            type = nullptr;
        }

        if (recursion)
        {
            delete recursion;
            recursion = nullptr;
        }
    }

    std::vector<std::pair<std::string, std::string>> TypedNameListNode::get_typed_names() const
    {
        if ((typed_names.size() > 0) && (type == nullptr))
        {
            throw std::runtime_error("internal typed_names is not empty while type is null");
        }

        std::vector<std::pair<std::string, std::string>> result;

        for (auto name : untyped_names)
        {
            result.push_back(std::make_pair(name->get_name(), "object"));
        }

        if (type)
        {
            auto base = type->get_type();

            for (auto name : typed_names)
            {
                result.push_back(std::make_pair(name->get_name(), base));
            }
        }

        if (recursion)
        {
            auto recursive_result = recursion->get_typed_names();
            result.insert(result.end(), recursive_result.begin(), recursive_result.end());
        }

        return result;
    }

    /* TypedVariableListNode */

    TypedVariableListNode::TypedVariableListNode(std::vector<VariableNode*>& untyped_names) :
        untyped_names(untyped_names),
        typed_names(),
        type(nullptr),
        recursion(nullptr)
    {
    }

    TypedVariableListNode::TypedVariableListNode(std::vector<VariableNode*>& typed_names, TypeNode* type, TypedVariableListNode* recursion) :
        untyped_names(),
        typed_names(typed_names),
        type(type),
        recursion(recursion)
    {
    }

    TypedVariableListNode::~TypedVariableListNode()
    {
        for (auto name : untyped_names)
        {
            delete name;
        }
        untyped_names.clear();

        for (auto name : typed_names)
        {
            delete name;
        }
        typed_names.clear();

        if (type)
        {
            delete type;
            type = nullptr;
        }

        if (recursion)
        {
            delete recursion;
            recursion = nullptr;
        }
    }

    std::vector<std::pair<std::string, std::string>> TypedVariableListNode::get_typed_variables() const
    {
        if ((typed_names.size() > 0) && (type == nullptr))
        {
            throw std::runtime_error("internal typed_names is not empty while type is null");
        }

        std::vector<std::pair<std::string, std::string>> result;

        for (auto name : untyped_names)
        {
            result.push_back(std::make_pair(name->get_variable(), "object"));
        }

        if (type)
        {
            auto base = type->get_type();

            for (auto name : typed_names)
            {
                result.push_back(std::make_pair(name->get_variable(), base));
            }
        }

        if (recursion)
        {
            auto recursive_result = recursion->get_typed_variables();
            result.insert(result.end(), recursive_result.begin(), recursive_result.end());
        }

        return result;
    }

    /* RequirementNode */

    RequirementNode::RequirementNode(formalism::Requirement requirement) : requirement(requirement) {}

    formalism::Requirement RequirementNode::get_requirement() const { return requirement; }

    /* RequirementListNode */

    RequirementListNode::RequirementListNode(std::vector<RequirementNode*>& requirements) : requirements(requirements) {}

    RequirementListNode::~RequirementListNode()
    {
        for (auto node : requirements)
        {
            delete node;
        }

        requirements.clear();
    }

    formalism::RequirementList RequirementListNode::get_requirements() const
    {
        formalism::RequirementList result;

        for (auto node : requirements)
        {
            result.push_back(node->get_requirement());
        }

        return result;
    }

    /* PredicateNode */

    PredicateNode::PredicateNode(NameNode* name, TypedVariableListNode* parameters) : name(name), parameters(parameters) {}

    PredicateNode::~PredicateNode()
    {
        if (name)
        {
            delete name;
            name = nullptr;
        }

        if (parameters)
        {
            delete parameters;
            parameters = nullptr;
        }
    }

    formalism::Predicate PredicateNode::get_predicate(const uint32_t id, const std::map<std::string, formalism::Type>& types) const
    {
        const auto predicate_name = name->get_name();
        const auto typed_variables = parameters->get_typed_variables();
        formalism::ObjectList predicate_parameters;

        uint32_t obj_id = 0;

        for (const auto& parameter : typed_variables)
        {
            const auto parameter_name = parameter.first;
            const auto parameter_type_handler = types.find(parameter.second);

            if (parameter_type_handler != types.end())
            {
                const auto parameter_type = parameter_type_handler->second;
                predicate_parameters.push_back(formalism::create_object(obj_id++, parameter_name, parameter_type));
            }
            else
            {
                throw std::invalid_argument("the type of parameter \"" + parameter_name + "\" is undefined");
            }
        }

        return formalism::create_predicate(id, name->get_name(), predicate_parameters);
    }

    /* PredicateListNode */

    PredicateListNode::PredicateListNode(std::vector<PredicateNode*>& predicates) : predicates(predicates) {}

    PredicateListNode::~PredicateListNode()
    {
        for (auto node : predicates)
        {
            delete node;
        }
        predicates.clear();
    }

    formalism::PredicateList PredicateListNode::get_predicates(const std::map<std::string, formalism::Type>& types) const
    {
        formalism::PredicateList result;
        uint32_t pred_id = 0;

        for (const auto predicate_node : predicates)
        {
            result.push_back(predicate_node->get_predicate(pred_id++, types));
        }

        return result;
    }

    /* FunctionDeclarationNode */

    FunctionDeclarationNode::FunctionDeclarationNode(PredicateNode* predicate, NameNode* type) : predicate(predicate), type(type) {}

    FunctionDeclarationNode::~FunctionDeclarationNode()
    {
        if (predicate)
        {
            delete predicate;
            predicate = nullptr;
        }

        if (type)
        {
            delete type;
            type = nullptr;
        }
    }

    /* FunctionDeclarationListNode */

    FunctionDeclarationListNode::FunctionDeclarationListNode(std::vector<FunctionDeclarationNode*>& functions) : functions(functions) {}

    FunctionDeclarationListNode::~FunctionDeclarationListNode()
    {
        for (auto node : functions)
        {
            delete node;
        }
        functions.clear();
    }

    /* FunctionNode */

    FunctionNode::FunctionNode(NameNode* op, AtomNode* first_operand, AtomNode* second_operand) :
        op(op),
        first_operand_atom(first_operand),
        second_operand_atom(second_operand),
        second_operand_value(0.0)
    {
    }

    FunctionNode::FunctionNode(NameNode* op, AtomNode* first_operand, double second_operand) :
        op(op),
        first_operand_atom(first_operand),
        second_operand_atom(nullptr),
        second_operand_value(second_operand)
    {
    }

    FunctionNode::~FunctionNode()
    {
        if (op)
        {
            delete op;
            op = nullptr;
        }

        if (first_operand_atom)
        {
            delete first_operand_atom;
            first_operand_atom = nullptr;
        }

        if (second_operand_atom)
        {
            delete second_operand_atom;
            second_operand_atom = nullptr;
        }
    }

    /* LiteralNode */

    LiteralNode::LiteralNode(bool negated, AtomNode* atom) : negated(negated), atom(atom) {}

    LiteralNode::~LiteralNode()
    {
        delete atom;
        atom = nullptr;
    }

    formalism::Literal LiteralNode::get_literal(const std::map<std::string, formalism::Parameter>& parameters,
                                                const std::map<std::string, formalism::Object>& constants,
                                                const std::map<std::string, formalism::Predicate>& predicates) const
    {
        const auto literal_atom = atom->get_atom(parameters, constants, predicates);
        return formalism::create_literal(literal_atom, negated);
    }

    /* LiteralListNode */

    LiteralListNode::LiteralListNode(LiteralNode* literal) : literals() { literals.push_back(literal); }

    LiteralListNode::LiteralListNode(std::vector<LiteralNode*>& literals) : literals(literals) {}

    LiteralListNode::~LiteralListNode()
    {
        for (auto atom : literals)
        {
            delete atom;
        }
        literals.clear();
    }

    formalism::LiteralList LiteralListNode::get_literals(const std::map<std::string, formalism::Parameter>& parameters,
                                                         const std::map<std::string, formalism::Object>& constants,
                                                         const std::map<std::string, formalism::Predicate>& predicates) const
    {
        formalism::LiteralList result;

        for (const auto node : literals)
        {
            result.push_back(node->get_literal(parameters, constants, predicates));
        }

        return result;
    }

    /* LiteralOrFunctionNode */

    LiteralOrFunctionNode::LiteralOrFunctionNode(LiteralNode* literal_node) : literal_node(literal_node), function_effect_node(nullptr) {}

    LiteralOrFunctionNode::LiteralOrFunctionNode(FunctionNode* function_effect_node) : literal_node(nullptr), function_effect_node(function_effect_node) {}

    LiteralOrFunctionNode::~LiteralOrFunctionNode()
    {
        if (literal_node)
        {
            delete literal_node;
            literal_node = nullptr;
        }

        if (function_effect_node)
        {
            delete function_effect_node;
            function_effect_node = nullptr;
        }
    }

    /* LiteralOrFunctionListNode */

    LiteralOrFunctionListNode::LiteralOrFunctionListNode(LiteralOrFunctionNode* literal_or_function) : literal_or_functions()
    {
        literal_or_functions.push_back(literal_or_function);
    }

    LiteralOrFunctionListNode::LiteralOrFunctionListNode(std::vector<LiteralOrFunctionNode*>& literal_or_functions) : literal_or_functions(literal_or_functions)
    {
    }

    LiteralOrFunctionListNode::~LiteralOrFunctionListNode()
    {
        for (auto literal_or_function : literal_or_functions)
        {
            delete literal_or_function;
        }
        literal_or_functions.clear();
    }

    formalism::LiteralList LiteralOrFunctionListNode::get_literals(const std::map<std::string, formalism::Parameter>& parameters,
                                                                   const std::map<std::string, formalism::Object>& constants,
                                                                   const std::map<std::string, formalism::Predicate>& predicates) const
    {
        formalism::LiteralList result;

        for (const auto node : literal_or_functions)
        {
            if (node->literal_node)
            {
                result.push_back(node->literal_node->get_literal(parameters, constants, predicates));
            }
        }

        return result;
    }

    /* ActionBodyNode */

    ActionBodyNode::ActionBodyNode(boost::optional<boost::fusion::vector<std::string, LiteralListNode*>>& precondition,
                                   boost::optional<boost::fusion::vector<std::string, LiteralOrFunctionListNode*>>& effect) :
        precondition(nullptr),
        effect(nullptr)
    {
        if (precondition)
        {
            this->precondition = boost::fusion::at_c<1>(precondition.value());
        }

        if (effect)
        {
            this->effect = boost::fusion::at_c<1>(effect.value());
        }
    }

    ActionBodyNode::ActionBodyNode(LiteralListNode* precondition, LiteralOrFunctionListNode* effect) : precondition(precondition), effect(effect) {}

    ActionBodyNode::~ActionBodyNode()
    {
        if (precondition)
        {
            delete precondition;
            precondition = nullptr;
        }

        if (effect)
        {
            delete effect;
            effect = nullptr;
        }
    }

    std::pair<formalism::LiteralList, formalism::LiteralList>
    ActionBodyNode::get_precondition_effect(const std::map<std::string, formalism::Parameter>& parameters,
                                            const std::map<std::string, formalism::Object>& constants,
                                            const std::map<std::string, formalism::Predicate>& predicates) const
    {
        const auto precondition_literals = precondition->get_literals(parameters, constants, predicates);
        const auto effect_literals = effect->get_literals(parameters, constants, predicates);
        return std::make_pair(precondition_literals, effect_literals);
    }

    /* ActionNode */

    ActionNode::ActionNode(NameNode* name, TypedVariableListNode* parameters, ActionBodyNode* body) : name(name), parameters(parameters), body(body) {}

    ActionNode::~ActionNode()
    {
        if (name)
        {
            delete name;
            name = nullptr;
        }

        if (parameters)
        {
            delete parameters;
            parameters = nullptr;
        }

        if (body)
        {
            delete body;
            body = nullptr;
        }
    }

    formalism::ActionSchema ActionNode::get_action(const std::map<std::string, formalism::Type>& types,
                                                   const std::map<std::string, formalism::Object>& constants,
                                                   const std::map<std::string, formalism::Predicate>& predicates) const
    {
        const auto action_name = name->get_name();

        formalism::ParameterList action_parameter_list;
        std::map<std::string, formalism::Parameter> action_parameter_map;
        const auto typed_parameters = parameters->get_typed_variables();
        uint32_t obj_id = 0;

        for (const auto& name_type : typed_parameters)
        {
            const auto parameter_name = name_type.first;
            const auto type_name = name_type.second;
            const auto type_handler = types.find(type_name);

            if (type_handler != types.end())
            {
                const auto parameter_type = type_handler->second;
                const auto parameter = formalism::create_object(obj_id++, parameter_name, parameter_type);
                action_parameter_list.push_back(parameter);
                action_parameter_map.insert(std::make_pair(parameter_name, parameter));
            }
            else
            {
                throw std::invalid_argument("the type of object \"" + parameter_name + "\" is undefined");
            }
        }

        const auto action_precondition_effect = body->get_precondition_effect(action_parameter_map, constants, predicates);

        return formalism::create_action_schema(action_name, action_parameter_list, action_precondition_effect.first, action_precondition_effect.second);
    }

    /* DomainNode */

    DomainNode::DomainNode(NameNode* name,
                           boost::optional<RequirementListNode*> requirements,
                           boost::optional<TypedNameListNode*> types,
                           boost::optional<TypedNameListNode*> constants,
                           boost::optional<PredicateListNode*> predicates,
                           boost::optional<FunctionDeclarationListNode*> functions,
                           std::vector<ActionNode*>& actions) :
        name(name),
        requirements(nullptr),
        types(nullptr),
        constants(nullptr),
        predicates(nullptr),
        actions(actions)
    {
        if (requirements)
        {
            this->requirements = requirements.value();
        }

        if (types)
        {
            this->types = types.value();
        }

        if (constants)
        {
            this->constants = constants.value();
        }

        if (predicates)
        {
            this->predicates = predicates.value();
        }

        if (functions)
        {
            this->functions = functions.value();
        }
    }

    DomainNode::~DomainNode()
    {
        if (name)
        {
            delete name;
            name = nullptr;
        }

        if (requirements)
        {
            delete requirements;
            requirements = nullptr;
        }

        if (types)
        {
            delete types;
            types = nullptr;
        }

        if (constants)
        {
            delete constants;
            constants = nullptr;
        }

        if (predicates)
        {
            delete predicates;
            predicates = nullptr;
        }

        for (auto node : actions)
        {
            delete node;
        }
        actions.clear();
    }

    std::map<std::string, formalism::Type> DomainNode::get_types() const
    {
        std::map<std::string, formalism::Type> result;

        result.insert(std::make_pair("object", formalism::create_type("object")));

        if (types)
        {
            for (const auto& name_base : types->get_typed_names())
            {
                const auto type_name = name_base.first;
                const auto base_name = name_base.second;

                if (type_name == "object")
                {
                    throw std::invalid_argument("type name \"object\" is reserved");
                }

                // get pointer to base type, or create one if it does not exist

                const auto base_type_handler = result.find(base_name);
                formalism::Type base_type;

                if (base_type_handler != result.end())
                {
                    base_type = base_type_handler->second;
                }
                else
                {
                    base_type = formalism::create_type(base_name);
                    result.insert(std::make_pair(base_name, base_type));
                }

                // other types might inherit from this type, and have created one, update it

                const auto name_type_handler = result.find(type_name);

                if (name_type_handler != result.end())
                {
                    // update existing type
                    auto type = name_type_handler->second;
                    type->base = base_type;
                }
                else
                {
                    // create new type
                    auto type = formalism::create_type(type_name, base_type);
                    result.insert(std::make_pair(type_name, type));
                }
            }
        }

        return result;
    }

    std::map<std::string, formalism::Object> DomainNode::get_constants(const std::map<std::string, formalism::Type>& types) const
    {
        std::map<std::string, formalism::Object> result;

        if (constants)
        {
            uint32_t obj_id = 0;

            for (const auto& name_type : constants->get_typed_names())
            {
                const auto object_name = name_type.first;
                const auto type_name = name_type.second;
                const auto type_handler = types.find(type_name);

                if (type_handler != types.end())
                {
                    const auto type = type_handler->second;
                    const auto object = formalism::create_object(obj_id++, object_name, type);
                    result.insert(std::make_pair(object_name, object));
                }
                else
                {
                    throw std::invalid_argument("the type of object \"" + object_name + "\" is undefined");
                }
            }
        }

        return result;
    }

    std::map<std::string, formalism::Predicate> DomainNode::get_predicates(const std::map<std::string, formalism::Type>& types) const
    {
        std::map<std::string, formalism::Predicate> result;

        if (predicates)
        {
            const auto predicate_list = predicates->get_predicates(types);

            for (const auto& predicate : predicate_list)
            {
                result.insert(std::make_pair(predicate->name, predicate));
            }
        }

        return result;
    }

    std::vector<formalism::ActionSchema> DomainNode::get_action_schemas(const std::map<std::string, formalism::Type>& types,
                                                                        const std::map<std::string, formalism::Object>& constants,
                                                                        const std::map<std::string, formalism::Predicate>& predicates) const
    {
        std::vector<formalism::ActionSchema> action_schemas;

        for (const auto node : this->actions)
        {
            action_schemas.push_back(node->get_action(types, constants, predicates));
        }

        return action_schemas;
    }

    // template<typename K, typename V>
    // std::vector<V> get_values(std::map<K, V> map) const
    // {
    //     std::vector<V> values;

    //     for (auto it = map.begin(); it != map.end(); ++it)
    //     {
    //         values.push_back(it->second);
    //     }

    //     return values;
    // }

    formalism::DomainDescription DomainNode::get_domain() const
    {
        const auto domain_name = name->get_name();
        const auto domain_requirements = (requirements != nullptr) ? (requirements->get_requirements()) : formalism::RequirementList();
        const auto domain_types = this->get_types();
        const auto domain_constants = this->get_constants(domain_types);
        const auto domain_predicates = this->get_predicates(domain_types);
        const auto domain_actions = this->get_action_schemas(domain_types, domain_constants, domain_predicates);

        return formalism::create_domain(domain_name,
                                        domain_requirements,
                                        get_values(domain_types),
                                        get_values(domain_constants),
                                        get_values(domain_predicates),
                                        domain_actions);
    }

    /* ProblemHeaderNode */

    ProblemHeaderNode::ProblemHeaderNode(NameNode* problem_name, NameNode* domain_name) : problem_name(problem_name), domain_name(domain_name) {}

    ProblemHeaderNode::~ProblemHeaderNode()
    {
        if (problem_name)
        {
            delete problem_name;
            problem_name = nullptr;
        }

        if (domain_name)
        {
            delete domain_name;
            domain_name = nullptr;
        }
    }

    std::string ProblemHeaderNode::get_problem_name() const { return problem_name->get_name(); }

    std::string ProblemHeaderNode::get_domain_name() const { return domain_name->get_name(); }

    /* ProblemNode */

    std::map<std::string, formalism::Object> ProblemNode::get_objects(uint32_t num_constants, const std::map<std::string, formalism::Type>& types) const
    {
        std::map<std::string, formalism::Object> result;

        if (objects)
        {
            uint32_t obj_id = num_constants;

            for (const auto& name_type : objects->get_typed_names())
            {
                const auto object_name = name_type.first;
                const auto type_name = name_type.second;
                const auto type_handler = types.find(type_name);

                if (type_handler != types.end())
                {
                    const auto type = type_handler->second;
                    const auto object = formalism::create_object(obj_id++, object_name, type);
                    result.insert(std::make_pair(object_name, object));
                }
                else
                {
                    throw std::invalid_argument("the type of object \"" + object_name + "\" is undefined");
                }
            }
        }

        return result;
    }

    template<typename K, typename V>
    std::vector<V> ProblemNode::get_values(std::map<K, V> map) const
    {
        std::vector<V> values;

        for (auto it = map.begin(); it != map.end(); ++it)
        {
            values.push_back(it->second);
        }

        return values;
    }

    formalism::AtomList ProblemNode::atoms_of(const formalism::LiteralList& literals) const
    {
        formalism::AtomList atoms;

        for (const auto& literal : literals)
        {
            atoms.push_back(literal->atom);
        }

        return atoms;
    }

    ProblemNode::ProblemNode(ProblemHeaderNode* problem_domain_name,
                             boost::optional<TypedNameListNode*> objects,
                             LiteralOrFunctionListNode* initial,
                             LiteralListNode* goal,
                             boost::optional<AtomNode*> metric) :
        problem_domain_name(problem_domain_name),
        objects(objects ? objects.value() : nullptr),
        initial(initial),
        goal(goal),
        metric(metric ? metric.value() : nullptr)
    {
    }

    ProblemNode::~ProblemNode()
    {
        if (problem_domain_name)
        {
            delete problem_domain_name;
            problem_domain_name = nullptr;
        }

        if (objects)
        {
            delete objects;
            objects = nullptr;
        }

        if (initial)
        {
            delete initial;
            initial = nullptr;
        }

        if (goal)
        {
            delete goal;
            goal = nullptr;
        }
    }

    formalism::ProblemDescription ProblemNode::get_problem(const std::string& filename, const formalism::DomainDescription& domain) const
    {
        const auto domain_name = problem_domain_name->get_domain_name();

        if (domain->name != domain_name)
        {
            throw std::invalid_argument("domain names do not match: \"" + domain_name + "\" and \"" + domain->name + "\"");
        }

        const auto type_map = domain->get_type_map();
        const auto predicate_map = domain->get_predicate_map();
        const auto constant_map = domain->get_constant_map();

        const auto problem_name = problem_domain_name->get_problem_name();
        const auto object_map = get_objects(constant_map.size(), type_map);

        std::vector<formalism::Literal> initial_list;
        for (const auto& node : initial->literal_or_functions)
        {
            if (node->literal_node)
            {
                initial_list.push_back(node->literal_node->get_literal(object_map, constant_map, predicate_map));
            }
        }

        const auto goal_list = goal->get_literals(object_map, constant_map, predicate_map);

        auto objects = get_values(object_map);
        objects.insert(objects.end(), domain->constants.begin(), domain->constants.end());

        return formalism::create_problem(problem_name + " (" + filename + ")", domain, objects, atoms_of(initial_list), goal_list);
    }
}  // namespace parsers
