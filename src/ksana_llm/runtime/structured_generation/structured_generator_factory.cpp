/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "structured_generator_factory.h"

#include <functional>
#include <mutex>
#include <stdexcept>

#include "ksana_llm/runtime/structured_generation/xgrammar/xgrammar_structured_generator_creator.h"

namespace ksana_llm {

std::shared_ptr<StructuredGeneratorInterface> StructuredGeneratorFactory::CreateGenerator(
    const StructuredGeneratorConfig& config) {
  std::lock_guard<std::mutex> lock(registry_mutex_);

  if (config.constraint_type == StructuredConstraintType::NONE) {
    return nullptr;
  }

  auto it = creator_registry_.find(config.constraint_type);
  if (it == creator_registry_.end()) {
    throw std::runtime_error("Unsupported constraint type: " +
                             std::to_string(static_cast<int>(config.constraint_type)));
  }

  return it->second->CreateGenerator(config);
}

bool StructuredGeneratorFactory::IsConstraintTypeSupported(StructuredConstraintType constraint_type) {
  std::lock_guard<std::mutex> lock(registry_mutex_);
  return creator_registry_.find(constraint_type) != creator_registry_.end();
}

std::vector<StructuredConstraintType> StructuredGeneratorFactory::GetSupportedConstraintTypes() {
  std::lock_guard<std::mutex> lock(registry_mutex_);

  std::vector<StructuredConstraintType> supported_types;
  for (const auto& pair : creator_registry_) {
    supported_types.push_back(pair.first);
  }

  return supported_types;
}

void StructuredGeneratorFactory::RegisterCreator(StructuredConstraintType constraint_type,
                                                 std::unique_ptr<GeneratorCreator> creator) {
  std::lock_guard<std::mutex> lock(registry_mutex_);
  creator_registry_[constraint_type] = std::move(creator);
}

}  // namespace ksana_llm
