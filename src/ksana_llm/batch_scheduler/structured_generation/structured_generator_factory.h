/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_scheduler/structured_generation/structured_generator_interface.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ksana_llm {

/*!
 * \\brief Abstract base class for generator creators
 *
 * Each Creator is responsible for creating generators for a specific constraint type.
 */
class GeneratorCreator {
 public:
  virtual ~GeneratorCreator() = default;

  virtual std::shared_ptr<StructuredGeneratorInterface> CreateGenerator(const StructuredGeneratorConfig& config) = 0;

  virtual StructuredConstraintType GetConstraintType() const = 0;
};

/*!
 * \\brief Factory class for creating structured generators.
 *
 * This factory creates appropriate structured generator instances based on
 * the constraint type and configuration using registered creator objects.
 */
class StructuredGeneratorFactory {
 public:
  StructuredGeneratorFactory(std::vector<std::string>& vocab, int vocab_size, std::vector<int>& stop_token_ids);
  std::shared_ptr<StructuredGeneratorInterface> CreateGenerator(const StructuredGeneratorConfig& config);

  bool IsConstraintTypeSupported(StructuredConstraintType constraint_type);

  std::vector<StructuredConstraintType> GetSupportedConstraintTypes();

  void RegisterCreator(StructuredConstraintType constraint_type, std::unique_ptr<GeneratorCreator> creator);

 private:
  StructuredGeneratorFactory();

  void InitializeRegistry();

  std::unordered_map<StructuredConstraintType, std::unique_ptr<GeneratorCreator>> creator_registry_;
  std::mutex registry_mutex_;
};

}  // namespace ksana_llm
