// Copyright 2024 Tencent Inc.  All rights reserved.
#pragma once

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ksana_llm {

/* NON_GENERATION_STATE: When in this state, a fixed text segment (obtained by parsing structured_output_regex) is added
 * after the current request's output_tokens. The request will be marked as "ContextDecode" and the state automatically
 * transitions to the default next state.
 *
 * GENERATION_STATE: When in this state, the request performs a ContextDecode/Decode inference once and determines if
 * the generated token can transition to the next state. If it can, the request transitions to the next state;
 * otherwise, it remains in the current state_id and continues generating the next token.
 *
 * FINAL_STATE: When the request reaches this state, it indicates that the structured output has been completed. The
 * request should be terminated and released.
 *
 * INVALID_STATE: An exceptional state that should never be reached at any time.
 */
enum FiniteStateType {
  NON_GENERATION_STATE,
  GENERATION_STATE,
  FINAL_STATE,
  INVALID_STATE,
};

/* We have a total of three types of state machine nodes:
 * The first type is the re2::Prog::Inst, which is internally implemented in re2. This state machine can have empty
 * edges and transition edges with characters as content between nodes. We cannot directly use the re2::Prog::Inst for
 * structured output. Therefore, we introduce the FiniteStateInstNode. We define that there can be empty edges and
 * transition edges with strings as content between nodes. After compression and merging, we obtain a graph composed of
 * FiniteStateInstNodes. Finally, we define the FiniteStateNode. We define that there are no empty edges between nodes.
 * The edge relationships store both strings and their corresponding token_ids.
 */

// The temporary FiniteState nodes converted from re2::Prog::Inst, the edges between nodes are associated using strings.
class FiniteStateInstNode {
 public:
  explicit FiniteStateInstNode(size_t state_id) {
    state_id_ = state_id;
    merge_id = state_id;
  }

  // Add an association edge and record that two points can be folded.
  void InsertEdge(size_t next_state_id);

  // Add an association edge.
  void InsertEdge(std::string& str, size_t next_state_id);

  // node's id
  size_t state_id_;

  // The state_id of the root node of this node.
  size_t merge_id;

  // The edges of this node.
  std::vector<std::pair<std::string, size_t>> edges;
};

class FiniteStateNode {
 public:
  FiniteStateNode(FiniteStateType state_type, size_t state_id) {
    state_type_ = state_type;
    state_id_ = state_id;
  }

  // Enter a token and output the state ID of the next state.
  size_t GetNextStateId(const int& token);

  // Use the default token to trans into next state.
  size_t GetNextStateId();

  // Retrieve the prompt for the edge (this function is only called in non-generation states).
  std::pair<std::string, size_t> GetEdge();

  // state_id_ represents the node's number in the entire FSM.
  size_t state_id_;

  // state_type_ indicates the Finite State Type of this node.
  FiniteStateType state_type_;

  // current state + token = next state (the constant string in edge, next state id)
  std::unordered_map<int, std::pair<std::string, size_t>> edges_map_;
};

class FiniteStateMachine {
 public:
  explicit FiniteStateMachine(std::string& str);

  // Build the finite state machine with pattern.
  void BuildFiniteStateMachine(const std::string& str);

  // Check if it is an end state.
  bool IsStopState(const size_t& state_id);

  // Use tokenizer to trans a string into a token list. And store in map.
  void TokenizeString2Tokens(const std::string& str, std::vector<int>& tokens);

  // Use special symbols and prefixes of different tokenzier. Tokenize and store the new str.
  void TokenizeWithSpecialSymbols(const std::string& str, std::vector<int>& token_list);

  // Check whether request need pop the last token when doing jump-forward
  void CheckFSMPopToken(const size_t& state_id, std::vector<int>& input_tokens);

  // Check if the string can be correctly parsed as UTF-8.
  bool IsUTF8Supported(const std::string& str);

  // Get the state_type of state_id.
  FiniteStateType GetStateType(const size_t& state_id);

  // Get the state in state_map_.
  std::shared_ptr<FiniteStateNode> GetState(size_t state_id) { return state_map_[state_id]; }

  // Return the token list of string.
  std::vector<int> GetStringTokens(const std::string& str) {
    if (string_tokens_map_.count(str)) {
      return string_tokens_map_[str];
    }
    return {};
  }

  std::unordered_map<size_t, std::shared_ptr<FiniteStateNode>> state_map_;

  std::unordered_map<std::string, std::vector<int>> string_tokens_map_;

 private:
  // Print the temporary finite state inst node graph.
  void DumpFiniteStateInstNodeGraph(std::vector<std::shared_ptr<FiniteStateInstNode>>& node_list,
                                    std::set<size_t>& valid_node_set);

  // Print the final finite state node graph.
  void DumpFiniteStateNodeGraph(const size_t& current_id, std::unordered_set<size_t>& node_set, std::string& str);
};

class FiniteStateMachineController {
 public:
  FiniteStateMachineController() {}

  // Create or get a FiniteStateMachine from a pattern
  std::shared_ptr<FiniteStateMachine> CreateOrGetFSM(std::string& str);

 private:
  std::unordered_map<std::string, std::shared_ptr<FiniteStateMachine>> fsm_map;
};

}  // namespace ksana_llm
