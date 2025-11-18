/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/structured_generation/structured_generator_interface.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ksana_llm/utils/config/schedule_config_parser.h"

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
  StructuredGeneratorFactory() {}
  std::shared_ptr<StructuredGeneratorInterface> CreateGenerator(const StructuredGeneratorConfig& config,
                                                                const bool enable_thinking);

  bool IsConstraintTypeSupported(StructuredConstraintType constraint_type);

  std::vector<StructuredConstraintType> GetSupportedConstraintTypes();

  void RegisterCreator(StructuredConstraintType constraint_type, std::unique_ptr<GeneratorCreator> creator);

  /*!
   * \brief Set the reasoning configuration
   * \param reasoning_config The reasoning configuration
   */
  void SetReasoningConfig(const ReasoningConfig& reasoning_config) { reasoning_config_ = reasoning_config; }

  /*!
   * \brief Get the reasoning configuration
   * \return The reasoning configuration
   */
  const ReasoningConfig& GetReasoningConfig() const { return reasoning_config_; }

 private:
  void InitializeRegistry();

  std::unordered_map<StructuredConstraintType, std::unique_ptr<GeneratorCreator>> creator_registry_;
  std::mutex registry_mutex_;
  ReasoningConfig reasoning_config_;  // Reasoning configuration
};

}  // namespace ksana_llm
