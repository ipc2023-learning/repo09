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


#include "../libraries/json.hpp"
#include "relational_neural_network.hpp"
#include "torch/torch.h"
#include "utils.hpp"

namespace models
{
    struct imembuf : std::streambuf
    {
        imembuf(char const* base, size_t size)
        {
            char* p(const_cast<char*>(base));
            this->setg(p, p, p + size);
        }
    };

    struct imemstream : virtual imembuf, std::istream
    {
        imemstream(char const* base, size_t size) : imembuf(base, size), std::istream(static_cast<std::streambuf*>(this)) {}
    };

    bool initialize_model(models::RelationalNeuralNetwork* out_model, nlohmann::json& hparams, torch::serialize::InputArchive& input_archive)
    {
        const auto type = hparams.contains("type") ? hparams.find("type")->get<std::string>() : "mpnn";

        if (type == "mpnn")
        {
            bool global_readout = false;
            DerivedPredicateList derived_predicates;

            if (hparams.contains("global_readout"))
            {
                global_readout = hparams.find("global_readout")->get<bool>();
            }

            if (hparams.contains("derived_predicates"))
            {
                derived_predicates = hparams.find("derived_predicates")->get<DerivedPredicateList>();
            }

            if (hparams.contains("features") && hparams.contains("layers") && hparams.contains("predicates"))
            {
                const auto num_features = hparams.find("features")->get<int32_t>();
                const auto num_layers = hparams.find("layers")->get<int32_t>();
                const auto predicates = hparams.find("predicates")->get<std::vector<std::pair<std::string, int32_t>>>();
                const auto maximum_smoothnesss = hparams.find("maximum_smoothness")->get<double>();

                try
                {
                    auto model = models::RelationalMessagePassingNeuralNetwork(predicates,
                                                                               derived_predicates,
                                                                               num_features,
                                                                               num_layers,
                                                                               global_readout,
                                                                               maximum_smoothnesss);
                    model->load(input_archive);
                    *out_model = models::RelationalNeuralNetwork(model);
                    return true;
                }
                catch (const std::exception& e)
                {
                    std::cout << "Error loading model: " << e.what() << std::endl;
                }
            }
        }
        else if (type == "transformer")
        {
            // TODO: Implement loading for Transformer.
        }
        else
        {
            std::cerr << "Error loading model: type \"" << type << "\" is not recognized" << std::endl;
        }

        return false;
    }

    void save_model(const fs::path& path, const models::RelationalNeuralNetwork& model)
    {
        auto model_path = path;
        auto hparams_path = path;

        if (path.has_extension())
        {
            model_path = model_path.replace_extension("pnn");
            hparams_path = hparams_path.replace_extension("hparams");
        }
        else
        {
            model_path = model_path.concat(".pnn");
            hparams_path = hparams_path.concat(".hparams");
        }

        // Save hyperparameters

        if (const auto mpnn = model.get_relational_neural_network())
        {
            nlohmann::json hparams_json;
            hparams_json["type"] = "mpnn";
            hparams_json["predicates"] = mpnn->predicates();
            hparams_json["derived_predicates"] = mpnn->derived_predicates();
            hparams_json["features"] = mpnn->hidden_size();
            hparams_json["layers"] = mpnn->number_of_layers();
            hparams_json["global_readout"] = mpnn->global_readout();
            hparams_json["maximum_smoothness"] = mpnn->maximum_smoothness();

            std::ofstream output_hparams(hparams_path);
            output_hparams << std::setw(4) << hparams_json << std::endl;

            // Save model

            torch::serialize::OutputArchive output_archive;
            mpnn->save(output_archive);
            output_archive.save_to(model_path.string());
        }
        else if (const auto transformer = model.get_relational_transformer())
        {
            nlohmann::json hparams_json;
            hparams_json["type"] = "transformer";
            hparams_json["predicates"] = transformer->predicates();
            hparams_json["derived_predicates"] = transformer->derived_predicates();
            hparams_json["embedding_size"] = transformer->embedding_size();
            hparams_json["num_layers"] = transformer->num_layers();
            hparams_json["num_identifiers"] = transformer->num_identifiers();
            hparams_json["num_attention_heads"] = transformer->num_attention_heads();

            std::ofstream output_hparams(hparams_path);
            output_hparams << std::setw(4) << hparams_json << std::endl;

            // Save model

            torch::serialize::OutputArchive output_archive;
            transformer->save(output_archive);
            output_archive.save_to(model_path.string());
        }
    }

    bool load_model(models::RelationalNeuralNetwork* loaded_model, const fs::path& path)
    {
        fs::path model_path = path;
        fs::path hparams_path = path;

        if (path.has_extension())
        {
            model_path = model_path.replace_extension("pnn");
            hparams_path = hparams_path.replace_extension("hparams");
        }
        else
        {
            model_path = model_path.concat(".pnn");
            hparams_path = hparams_path.concat(".hparams");
        }

        if (fs::exists(model_path) && fs::exists(hparams_path))
        {
            try
            {
                std::ifstream input_hparams(hparams_path);
                nlohmann::json hparams_json;
                input_hparams >> hparams_json;

                torch::serialize::InputArchive input_archive;
                input_archive.load_from(model_path.string(), torch::kCPU);

                return initialize_model(loaded_model, hparams_json, input_archive);
            }
            catch (const std::exception& e)
            {
                std::cout << "Error loading model: " << e.what() << std::endl;
            }
        }

        return false;
    }
}  // namespace models
