/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <any>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ksana_llm/models/weight_method/common_method.h"
#include "ksana_llm/models/weight_method/w4a8_awq_method.h"

namespace ksana_llm {

class FunctionHelper {
 private:
  std::function<std::any(std::vector<std::any>&)> func_;

 public:
  // 注册任意签名的成员函数
  template <typename ClassType, typename ReturnType, typename... Args>
  void Add(ClassType* obj, ReturnType (ClassType::*func)(Args...)) {
    func_ = [obj, func](std::vector<std::any>& args) -> std::any {
      return callHelper(obj, func, args, std::index_sequence_for<Args...>{});
    };
  }

  // 调用函数
  template <typename ReturnType, typename... Args>
  ReturnType Run(Args&&... args) {
    if (!func_) {
      throw std::runtime_error("Function not registered");
    }
    std::vector<std::any> argVec = {std::any(std::ref(args))...};
    std::any result = func_(argVec);
    return std::any_cast<ReturnType>(result);
  }

 private:
  template <typename ClassType, typename ReturnType, typename... Args, std::size_t... Is>
  static std::any callHelper(ClassType* obj, ReturnType (ClassType::*func)(Args...), std::vector<std::any>& args,
                             std::index_sequence<Is...>) {
    return (obj->*func)(std::any_cast<std::reference_wrapper<std::remove_reference_t<Args>>>(args[Is]).get()...);
  }
};

class WeightMethod {
 public:
  using RegistryType = std::unordered_map<std::string, std::unordered_map<std::string, FunctionHelper>>;

  WeightMethod(std::shared_ptr<CommonModelWeightLoader> common_weight_loader, int tp);

  std::shared_ptr<CommonMethod> GetCommonMethod() const { return common_method_; }
  std::shared_ptr<W4A8AWQMethod> GetW4A8AWQMethod() const { return w4a8_awq_method_; }
  RegistryType& GetRegistry() { return registry_; }

 public:
  // 两层 map 结构: registry_[name][stage] 存储 FunctionHelper 对象
  RegistryType registry_;

 private:
  std::shared_ptr<CommonMethod> common_method_;
  std::shared_ptr<W4A8AWQMethod> w4a8_awq_method_;
};

}  // namespace ksana_llm