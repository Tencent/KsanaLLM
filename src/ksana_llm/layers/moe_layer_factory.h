/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class MoeLayerFactory {
 public:
  typedef std::shared_ptr<BaseLayer> (MoeLayerFactory::*BuildLayerFunc)();
  MoeLayerFactory(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                  std::shared_ptr<Context> context);

  ~MoeLayerFactory() {}

  template <typename ClassT>
  std::shared_ptr<BaseLayer> BuildLayer() {
    return std::make_shared<ClassT>();
  }

  std::shared_ptr<BaseLayer> AutoCreateMoeLayer(std::shared_ptr<BaseWeight> base_weight,
                                                std::vector<std::string> weight_names, DataType weight_type,
                                                DataType input_type, DataType output_type,
                                                const std::vector<std::any>& init_params);

  std::shared_ptr<BaseLayer> CreateLayer(DataType weight_type, DataType input_type, DataType output_type,
                                         const std::vector<std::any>& init_params, QuantMode quant_mode,
                                         GroupQuantBackend backend);

 private:
  std::shared_ptr<Context> context_;
  int rank_;
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;

  // std::map<std::tuple<weight_type, input_type, output_type, quant_mode, backend>, BuildLayerFunc>
  std::map<std::tuple<DataType, DataType, DataType, QuantMode, GroupQuantBackend>, BuildLayerFunc> builder_map_;
};

}  // namespace ksana_llm
