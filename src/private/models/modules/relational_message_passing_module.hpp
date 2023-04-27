#if !defined(RELATIONAL_MESSAGE_PASSING_MODULE_HPP_)
#define RELATIONAL_MESSAGE_PASSING_MODULE_HPP_

#include "mlp_module.hpp"
#include "relation_module.hpp"
#include "torch/torch.h"

class RelationalMessagePassingModuleImpl : public torch::nn::Module
{
  private:
    int32_t hidden_size_;
    torch::nn::ModuleList relation_modules_;
    MLPModule update_module_;
    double maximum_smoothness_;
    torch::Tensor dummy_;

  public:
    RelationalMessagePassingModuleImpl() : hidden_size_(0), relation_modules_(), update_module_(), maximum_smoothness_(0), dummy_() {}

    RelationalMessagePassingModuleImpl(const std::vector<std::pair<int32_t, int32_t>>& id_arities, const int32_t hidden_size, const double maximum_smoothness);

    torch::Tensor forward(const torch::Tensor& object_embeddings, const std::map<int32_t, torch::Tensor>& relations);

  private:
    torch::Device device() const { return this->dummy_.device(); };
};

TORCH_MODULE(RelationalMessagePassingModule);

#endif  // RELATIONAL_MESSAGE_PASSING_MODULE_HPP_
