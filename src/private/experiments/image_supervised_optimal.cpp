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


#include "../datasets/balanced_dataset.hpp"
#include "../datasets/random_dataset.hpp"
#include "../models/convolutional_neural_network.hpp"
#include "image_supervised_optimal.hpp"

#include <random>

namespace experiments
{
    torch::Tensor to_image_blocks(const formalism::State& state, const formalism::ProblemDescription& problem)
    {
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        auto rng = std::default_random_engine(seed);

        const int32_t WIDTH = 16;
        const int32_t HEIGHT = 16;
        const int32_t DEPTH = 3;

        const std::vector<std::tuple<double, double, double>> COLORS({
            std::make_tuple(1.0, 0.0, 0.0),
            std::make_tuple(0.0, 1.0, 0.0),
            std::make_tuple(0.0, 0.0, 1.0),
            std::make_tuple(1.0, 1.0, 0.0),
            std::make_tuple(1.0, 0.0, 1.0),
            std::make_tuple(0.0, 1.0, 1.0),
            std::make_tuple(1.0, 1.0, 1.0),
        });

        const auto predicate_map = problem->domain->get_predicate_map();
        const auto state_atoms_by_predicate = state->get_atoms_grouped_by_predicate();
        const auto goal_atoms_by_predicate = formalism::create_state(formalism::as_atoms(problem->goal), problem)->get_atoms_grouped_by_predicate();
        const auto blocks = problem->objects;

        // Assign goal blocks as one color, other objects as another.
        const auto goal_blocks = formalism::get_objects(goal_atoms_by_predicate.at(predicate_map.at("clear")), 0);
        std::vector<std::size_t> color_assignment(blocks.size());

        for (std::size_t index = 0; index < blocks.size(); ++index)
        {
            const auto& block = blocks[index];
            if (std::count(goal_blocks.begin(), goal_blocks.end(), block))
            {
                color_assignment[index] = 0;
            }
            else
            {
                color_assignment[index] = 1;
            }
        }

        // Assign an unique, random, color to each block
        // std::vector<std::size_t> color_assignment(blocks.size());
        // std::iota(color_assignment.begin(), color_assignment.end(), 0);
        // // std::shuffle(color_assignment.begin(), color_assignment.end(), rng);

        std::map<formalism::Object, std::size_t> block_colors;
        for (std::size_t index = 0; index < blocks.size(); ++index)
        {
            block_colors.insert(std::make_pair(blocks[index], color_assignment[index]));
        }

        // Assign a random column to each block on the table

        std::vector<std::size_t> column_assignment(WIDTH);
        std::iota(column_assignment.begin(), column_assignment.end(), 0);
        std::shuffle(column_assignment.begin(), column_assignment.end(), rng);

        const auto on_table_objects = formalism::get_objects(state_atoms_by_predicate.at(predicate_map.at("ontable")), 0);
        std::map<formalism::Object, std::size_t> column;
        std::map<formalism::Object, std::size_t> row;

        for (std::size_t index = 0; index < on_table_objects.size(); ++index)
        {
            // Assign columns from left to right.
            // The row HEIGHT - 1 is reserved for HOLDING and CLEAR in the goal.

            const auto& obj = on_table_objects[index];
            column.insert(std::make_pair(obj, column_assignment[index]));
            row.insert(std::make_pair(obj, HEIGHT - 2));
        }

        const auto& predicate_on = predicate_map.at("on");
        if (state_atoms_by_predicate.find(predicate_on) != state_atoms_by_predicate.end())
        {
            const auto on_atoms = state_atoms_by_predicate.at(predicate_on);
            auto on_above_objects = formalism::get_objects(on_atoms, 0);
            auto on_below_objects = formalism::get_objects(on_atoms, 1);

            while (on_above_objects.size() > 0)
            {
                for (int index = on_above_objects.size() - 1; index >= 0; index--)
                {
                    const auto& above_object = on_above_objects[index];
                    const auto& below_object = on_below_objects[index];

                    if (column.find(below_object) != column.end())
                    {
                        const auto x = column[below_object];
                        const auto y = row[below_object] - 1;

                        column.insert(std::make_pair(above_object, x));
                        row.insert(std::make_pair(above_object, y));

                        on_above_objects.erase(on_above_objects.begin() + index);
                        on_below_objects.erase(on_below_objects.begin() + index);
                    }
                }
            }
        }

        // Encode column and row as an image with three channels (RGB)

        const auto IMAGE_SIZE = DEPTH * HEIGHT * WIDTH;
        const auto CHANNEL_OFFSET = HEIGHT * WIDTH;
        const auto HEIGHT_OFFSET = WIDTH;
        double image[IMAGE_SIZE];
        std::fill_n(image, IMAGE_SIZE, 0.0);

        for (const auto& block : blocks)
        {
            if (column.find(block) != column.end())
            {
                const auto color_index = block_colors[block];
                const auto color = COLORS[color_index];
                const auto x = column[block];
                const auto y = row[block];

                image[0 * CHANNEL_OFFSET + y * HEIGHT_OFFSET + x] = std::get<0>(color);
                image[1 * CHANNEL_OFFSET + y * HEIGHT_OFFSET + x] = std::get<1>(color);
                image[2 * CHANNEL_OFFSET + y * HEIGHT_OFFSET + x] = std::get<2>(color);
            }
        }

        const auto predicate_holding = predicate_map.at("holding");
        if (state_atoms_by_predicate.find(predicate_holding) != state_atoms_by_predicate.end())
        {
            const auto holding_object = formalism::get_objects(state_atoms_by_predicate.at(predicate_holding), 0).at(0);
            const auto color_index = block_colors[holding_object];
            const auto color = COLORS[color_index];

            // We don't have to fill the entire channel, but we do for clarity.

            image[0 * CHANNEL_OFFSET + (HEIGHT - 1) * HEIGHT_OFFSET + 0] = std::get<0>(color);
            image[1 * CHANNEL_OFFSET + (HEIGHT - 1) * HEIGHT_OFFSET + 0] = std::get<1>(color);
            image[2 * CHANNEL_OFFSET + (HEIGHT - 1) * HEIGHT_OFFSET + 0] = std::get<2>(color);
        }

        for (std::size_t index = 0; index < goal_blocks.size(); ++index)
        {
            const auto block = goal_blocks[index];
            const auto color_index = block_colors[block];
            const auto color = COLORS[color_index];

            image[0 * CHANNEL_OFFSET + (HEIGHT - 1) * HEIGHT_OFFSET + (1 + index)] = std::get<0>(color);
            image[1 * CHANNEL_OFFSET + (HEIGHT - 1) * HEIGHT_OFFSET + (1 + index)] = std::get<1>(color);
            image[2 * CHANNEL_OFFSET + (HEIGHT - 1) * HEIGHT_OFFSET + (1 + index)] = std::get<2>(color);
        }

        const auto image_tensor = torch::from_blob(image, { DEPTH, HEIGHT, WIDTH }, torch::TensorOptions().dtype(torch::kFloat64)).to(torch::kFloat32);
        return image_tensor;
    }

    torch::Tensor to_image(const formalism::State& state, const formalism::ProblemDescription& problem)
    {
        const auto& domain_name = problem->domain->name;

        if (domain_name.find("blocks") != std::string::npos)
        {
            return to_image_blocks(state, problem);
        }
        else
        {
            throw std::runtime_error("to_image is not implemented for domain: " + domain_name);
        }
    }

    void ImageSupervisedOptimal::fit(models::RelationalNeuralNetwork& ignored_model,  // Ignore model for now, create our own.
                                     const planners::StateSpaceList& training_state_spaces,
                                     const planners::StateSpaceList& validation_state_spaces)
    {
        const datasets::BalancedDataset training_set(training_state_spaces, false);
        const datasets::RandomDataset validation_set(validation_state_spaces, false);

        const auto device = ignored_model.device();

        const int32_t WIDTH = 16;
        const int32_t HEIGHT = 16;
        const int32_t DEPTH = 3;

        models::ConvolutionalNeuralNetwork model(HEIGHT, WIDTH, DEPTH, 16, 32);
        model->to(device);
        torch::optim::Adam optimizer(model->parameters(), learning_rate_);

        const uint32_t num_train_batches = (training_set.size() / batch_size_) + ((training_set.size() % batch_size_) > 0 ? 1 : 0);
        const uint32_t num_val_batches = (validation_set.size() / batch_size_) + ((validation_set.size() % batch_size_) > 0 ? 1 : 0);

        for (uint32_t epoch = 0; epoch < max_epochs_; ++epoch)
        {
            model->train();

            for (std::size_t batch_index = 0; batch_index < num_train_batches; ++batch_index)
            {
                model->zero_grad();
                const auto batch_states = training_set.get_range(batch_index * batch_size_, batch_size_);
                std::vector<torch::Tensor> batch_images;
                std::vector<double> batch_targets;

                for (const auto& sample : batch_states)
                {
                    const auto state_image = to_image(sample.first, sample.second->problem);
                    batch_images.push_back(state_image);
                    batch_targets.push_back(sample.second->get_distance_to_goal_state(sample.first));
                }

                const auto input = torch::stack(batch_images).to(device);
                const auto target = torch::tensor(batch_targets).to(device);
                const auto output = model->forward(input);
                const auto loss = (output.view(-1) - target.view(-1)).abs().mean();
                loss.backward();
                optimizer.step();

                std::cout << "[" << epoch + 1 << "] Train loss: " << loss.item<double>() << std::endl;
            }

            {  // Disable computation of gradients
                torch::NoGradGuard no_grad;
                model->eval();
                double total_validation_loss = 0;
                double total_validation_samples = 0;

                for (std::size_t batch_index = 0; batch_index < num_val_batches; ++batch_index)
                {
                    const auto batch_states = validation_set.get_range(batch_index * batch_size_, batch_size_);
                    std::vector<torch::Tensor> batch_images;
                    std::vector<double> batch_targets;

                    for (const auto& sample : batch_states)
                    {
                        const auto state_image = to_image(sample.first, sample.second->problem);
                        batch_images.push_back(state_image);
                        batch_targets.push_back(sample.second->get_distance_to_goal_state(sample.first));
                    }

                    const auto input = torch::stack(batch_images).to(device);
                    const auto target = torch::tensor(batch_targets).to(device);
                    const auto output = model->forward(input);
                    const auto loss = (output.view(-1) - target.view(-1)).abs().sum();

                    total_validation_loss += loss.item<double>();
                    total_validation_samples += batch_images.size();
                }

                std::cout << "[" << epoch + 1 << "] Validation loss: " << (total_validation_loss / total_validation_samples) << std::endl;
            }
        }
    }
}  // namespace experiments
