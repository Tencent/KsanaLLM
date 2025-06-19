// Copyright 2024 Tencent Inc.  All rights reserved.
#include "finite_state_machine.h"

#include <codecvt>
#include <iostream>
#include <locale>
#include <queue>

#include "absl/strings/str_join.h"
#include "re2/prog.h"
#include "re2/re2.h"
#include "re2/regexp.h"

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/string_utils.h"
#include "ksana_llm/utils/tokenizer.h"

namespace ksana_llm {

constexpr int kDefaultToken = -1;

size_t FiniteStateNode::GetNextStateId(const int& token) {
  /*
   * @brief Returns the next state identifier based on the input token and current state.
   */
  size_t next_state_id = state_id_;
  if (edges_map_.find(token) != edges_map_.end()) {
    next_state_id = edges_map_[token].second;
  }
  KLLM_LOG_DEBUG << fmt::format("S{} + token {} = S{}", state_id_, token, next_state_id);
  return next_state_id;
}

size_t FiniteStateNode::GetNextStateId() {
  if (edges_map_.find(kDefaultToken) == edges_map_.end()) {
    KLLM_LOG_DEBUG << fmt::format("S{} do not has an edge with the default token {}", state_id_, kDefaultToken);
    return state_id_;
  }
  KLLM_LOG_DEBUG << fmt::format("S{} + token {} = S{}", state_id_, -1, edges_map_[kDefaultToken].second);
  return edges_map_[kDefaultToken].second;
}

std::pair<std::string, size_t> FiniteStateNode::GetEdge() {
  if (edges_map_.find(kDefaultToken) != edges_map_.end()) {
    return edges_map_[kDefaultToken];
  }
  return std::pair<std::string, size_t>();
}

void FiniteStateMachine::TokenizeString2Tokens(const std::string& str, std::vector<int>& tokens) {
  // Use RequestPacker to convert a string into a token sequence and store it in string_tokens_map_.
  if (str.empty()) {
    tokens = {kDefaultToken};
    return;
  }
  if (string_tokens_map_.find(str) != string_tokens_map_.end()) {
    tokens = string_tokens_map_[str];
    return;
  }

  tokens.clear();
  if (IsUTF8Supported(str)) {
    Singleton<Tokenizer>::GetInstance()->Encode(str, tokens, /* add_special_tokens */ false);
    string_tokens_map_[str] = tokens;
  }
}

bool FiniteStateMachine::IsUTF8Supported(const std::string& str) {
  // Some complex regular expressions, such as [^"], indicate that any character except double quotes is acceptable.
  // When constructing a state machine, these characters are concatenated as strings, which may result in the presence
  // of UTF-8 characters that cannot be parsed, such as 0xc2, causing program exceptions and interruptions.
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  try {
    std::wstring wstr = converter.from_bytes(str);
    return true;
  } catch (...) {
    // range_error or other errors.
    return false;
  }
}

void FiniteStateInstNode::InsertEdge(std::string& str, size_t next_state_id) {
  // Add a conversion edge with a constant string.
  edges.push_back(std::make_pair(str, next_state_id));
}

void FiniteStateInstNode::InsertEdge(size_t next_state_id) {
  // Add a conversion edge without a constant string, which means these two nodes can be collapsed in the subsequent
  // compression step.
  merge_id = std::min(merge_id, next_state_id);
  edges.emplace_back("", next_state_id);
}

void FiniteStateMachine::TokenizeWithSpecialSymbols(const std::string& str, std::vector<int>& token_list) {
  // Special symbols and prefixes of SentencePiece tokenizer and BPE tokenizer.
  std::vector<int> tokens;
  TokenizeString2Tokens(str, tokens);
  if (!tokens.empty()) {
    token_list.push_back(tokens[0]);
  }
  TokenizeString2Tokens(fmt::format("\u2581{}", str), tokens);
  if (!tokens.empty()) {
    token_list.push_back(tokens[0]);
  }
  TokenizeString2Tokens(fmt::format("\u0120{}", str), tokens);
  if (!tokens.empty()) {
    token_list.push_back(tokens[0]);
  }
}

void FiniteStateMachine::DumpFiniteStateInstNodeGraph(std::vector<std::shared_ptr<FiniteStateInstNode>>& node_list,
                                                      std::set<size_t>& valid_node_set) {
  // For Debug only: print current finite state node graph
  for (size_t id : valid_node_set) {
    for (auto& p : node_list[id]->edges) {
      KLLM_LOG_DEBUG << fmt::format("S{} + {} = S{}", id, p.first, p.second);
    }
  }
}

void ReplaceSpecialCharacters(std::string& input) {
  std::string target = ":";
  std::string replacement = "COLON";
  std::string escapeSequence = "\\n";

  size_t pos = input.find(target);
  while (pos != std::string::npos) {
    input.replace(pos, target.length(), replacement);
    pos = input.find(target, pos + replacement.length());
  }

  pos = input.find("\n");
  while (pos != std::string::npos) {
    input.replace(pos, 1, escapeSequence);
    pos = input.find("\n", pos + escapeSequence.length());
  }

  pos = input.find("\t");
  while (pos != std::string::npos) {
    input.replace(pos, 1, "\\t");
    pos = input.find("\t", pos + 2);
  }
}

void FiniteStateMachine::DumpFiniteStateNodeGraph(const size_t& current_id, std::unordered_set<size_t>& node_set,
                                                  std::string& str) {
  if (node_set.find(current_id) != node_set.end() || state_map_.find(current_id) == state_map_.end()) {
    return;
  }
  node_set.insert(current_id);
  if (str.empty()) {
    str = fmt::format("stateDiagram-v2\n    [*] --> S{}\n", current_id);
  }
  std::unordered_set<std::string> tmp_edge_set;
  for (auto& edge : state_map_[current_id]->edges_map_) {
    std::string edge_str = edge.second.first;
    ReplaceSpecialCharacters(edge_str);
    std::string edge_format = fmt::format("    S{} --> S{} : {}\n", current_id, edge.second.second, edge_str);
    if (tmp_edge_set.find(edge_format) == tmp_edge_set.end()) {
      str += edge_format;
      tmp_edge_set.insert(edge_format);
    }
  }
  for (auto& edge : state_map_[current_id]->edges_map_) {
    DumpFiniteStateNodeGraph(edge.second.second, node_set, str);
  }
  if (state_map_[current_id]->edges_map_.empty()) {
    str += fmt::format("    S{} --> [*] : finish!\n", current_id);
  }
}

void FiniteStateMachine::BuildFiniteStateMachine(const std::string& str) {
  KLLM_LOG_DEBUG << fmt::format("Begin to Build Finite State Machine: pattern = {}", str);
  std::string pattern = str;
  re2::RE2 re(pattern);
  re2::Prog* prog = re.Regexp()->CompileToProg(0);

  // Step 1: convert from an RE2 state machine to a FiniteStateInstNode graph.
  std::vector<std::shared_ptr<FiniteStateInstNode>> node_list(prog->size() + 1);

  // Step 1.1: Initialize the Finite State Inst Node List Space.
  for (int id = 0; id <= prog->size(); ++id) {
    node_list[id] = std::make_shared<FiniteStateInstNode>(id);
  }
  size_t finish_node_id = static_cast<size_t>(prog->size());

  // Step 1.2: Iterate through all nodes to construct an unpruned FiniteStateInst graph.
  for (int id = 0; id < prog->size(); ++id) {
    re2::Prog::Inst* ins = prog->inst(id);
    switch (ins->opcode()) {
      case re2::kInstCapture:
        for (int next_id = id + 1; next_id < prog->size(); ++next_id) {
          node_list[id]->InsertEdge(next_id);
          node_list[next_id]->InsertEdge(id);
          re2::Prog::Inst* next_ins = prog->inst(next_id);
          if (next_ins->last()) {
            break;
          }
        }
        break;
      case re2::kInstNop:
        node_list[id]->InsertEdge(ins->out());
        node_list[ins->out()]->InsertEdge(id);
        break;
      case re2::kInstMatch:
        node_list[id]->InsertEdge(finish_node_id);
        node_list[finish_node_id]->InsertEdge(id);
        break;
      case re2::kInstByteRange:
        // re2::Prog::Inst::last(): When the value is False, it indicates that there is an empty edge from this State to
        // the next State.
        if (!ins->last()) {
          for (int next_id = id + 1; next_id < prog->size(); ++next_id) {
            re2::Prog::Inst* next_ins = prog->inst(next_id);
            node_list[id]->InsertEdge(next_id);
            node_list[next_id]->InsertEdge(id);
            if (next_ins->last()) {
              break;
            }
          }
        }
        for (int idx = ins->lo(); idx <= ins->hi(); ++idx) {
          std::string edge = std::string(1, static_cast<char>(idx));
          node_list[id]->InsertEdge(edge, ins->out());
        }
        break;
      default:
        break;
    }
  }
  KLLM_LOG_DEBUG << fmt::format(
      "BuildFiniteStateMachine Step 1: FiniteStateInstNode Graph(Sparse) Build Success! Node list size = {}",
      node_list.size());

  // Step 2: Compress the FiniteStateInstNode graph.
  // Step 2.1: Count all reachable nodes.
  std::queue<size_t> valid_node_queue;
  std::set<size_t> valid_node_set;
  std::unordered_set<size_t> invalid_node_list;
  // By default, the starting state index in the RE2 state machine is 1.
  valid_node_queue.push(1);
  while (!valid_node_queue.empty()) {
    size_t node_id = valid_node_queue.front();
    valid_node_queue.pop();
    if (valid_node_set.find(node_id) != valid_node_set.end()) {
      continue;
    }
    valid_node_set.insert(node_id);
    for (auto& child : node_list[node_id]->edges) {
      valid_node_queue.push(child.second);
    }
  }
  KLLM_LOG_DEBUG << fmt::format("BuildFiniteStateMachine Step 2.1: Valid Node Size = {}", valid_node_set.size());
  DumpFiniteStateInstNodeGraph(node_list, valid_node_set);

  // Step 2.2: Delete all empty edges and mergeable nodes.
  // Use merge_id to mark the root node to which each node belongs.
  for (size_t id : valid_node_set) {
    size_t idx = id;
    while (node_list[idx]->merge_id != idx) {
      idx = node_list[idx]->merge_id;
    }
    node_list[id]->merge_id = idx;
  }
  for (size_t id : valid_node_set) {
    if (node_list[id]->merge_id != id) {
      invalid_node_list.insert(id);
    }
    // Pass all child nodes of the current node to the root node.
    for (auto& child : node_list[id]->edges) {
      if (node_list[id]->merge_id == node_list[child.second]->merge_id) {
        // Ignore edges that point to the node itself.
        continue;
      }
      if (valid_node_set.find(child.second) == valid_node_set.end()) {
        continue;
      }
      size_t src_id = node_list[id]->merge_id;
      size_t dst_id = node_list[child.second]->merge_id;
      node_list[src_id]->InsertEdge(child.first, dst_id);
    }
    // Remove all empty edges from the current node.
    for (auto it = node_list[id]->edges.begin(); it != node_list[id]->edges.end();) {
      auto& child = *it;
      if (child.first.empty()) {
        node_list[id]->edges.erase(it);
      } else {
        ++it;
      }
    }
  }
  for (size_t id : invalid_node_list) {
    valid_node_set.erase(id);
    KLLM_LOG_DEBUG << fmt::format("Delete node S{} due to the presence of mergeable empty edges.", id);
  }
  invalid_node_list.clear();
  KLLM_LOG_DEBUG << fmt::format("BuildFiniteStateMachine Step 2.2: Empty Node Merge. Valid Node Size = {}",
                                valid_node_set.size());
  DumpFiniteStateInstNodeGraph(node_list, valid_node_set);

  // Step 2.3: Merge all equivalent edges.
  for (size_t id : valid_node_set) {
    if (node_list[id]->edges.size() <= 1) {
      // Skip directly when a node has only one edge.
      continue;
    }
    std::vector<std::pair<std::string, size_t>> new_edges;
    std::unordered_map<std::string, size_t> child_string_map;
    for (auto& child : node_list[id]->edges) {
      if (child_string_map.find(child.first) == child_string_map.end()) {
        // When the string first appears, store it in a map.
        child_string_map[child.first] = child.second;
        new_edges.push_back(child);
      } else if (child_string_map[child.first] != child.second) {
        // When the string reappears, merge the duplicate edges with the existing ones.
        size_t valid_state_id = child_string_map[child.first];
        node_list[valid_state_id]->edges.insert(node_list[valid_state_id]->edges.end(),
                                                node_list[child.second]->edges.begin(),
                                                node_list[child.second]->edges.end());
      }
    }
    if (new_edges != node_list[id]->edges) {
      node_list[id]->edges = new_edges;
    }
  }
  KLLM_LOG_DEBUG << fmt::format("BuildFiniteStateMachine Step 2.3: Merge equivalent edges. Valid Node Size = {}",
                                valid_node_set.size());
  // DumpFiniteStateInstNodeGraph(node_list, valid_node_set);

  // Step 2.4: Merge all individual edges of the form A->B->C, where B has exactly one input and exactly one output.
  // Calculate the in-degree and out-degree of all nodes.
  std::vector<int> input_degree(node_list.size(), 0);
  std::vector<int> output_degree(node_list.size(), 0);
  for (size_t id : valid_node_set) {
    output_degree[id] = node_list[id]->edges.size();
    for (auto& child : node_list[id]->edges) {
      input_degree[child.second]++;
    }
  }
  for (size_t id : valid_node_set) {
    for (auto it = node_list[id]->edges.begin(); it != node_list[id]->edges.end();) {
      auto& child = *it;
      // The in-degree of a child node is 1, the out-degree is 1, and the edge leading to the child node is not "", and
      // the edge pointed by the child node is also not "".
      if (input_degree[child.second] == 1 && child.first != "*" && output_degree[child.second] == 1 &&
          node_list[child.second]->edges[0].first != "*" && invalid_node_list.find(id) == invalid_node_list.end()) {
        std::string origin_edge_str = child.first;
        size_t origin_next_state_id = child.second;
        // The skipped nodes can be marked as invalid nodes.
        invalid_node_list.insert(child.second);
        child.first = fmt::format("{}{}", child.first, node_list[child.second]->edges[0].first);
        child.second = node_list[child.second]->edges[0].second;
        KLLM_LOG_DEBUG << fmt::format("Update S{} + {} = S{} ==> + {} = S{}", id, origin_edge_str, origin_next_state_id,
                                      child.first, child.second);
      } else {
        ++it;
      }
    }
  }
  for (size_t id : invalid_node_list) {
    valid_node_set.erase(id);
    KLLM_LOG_DEBUG << fmt::format("Delete node S{} due to the presence of mergeable individual edges.", id);
  }
  invalid_node_list.clear();
  KLLM_LOG_DEBUG << fmt::format("BuildFiniteStateMachine Step 2.4: Individual Edge Merge. Valid Node Size = {}",
                                valid_node_set.size());
  // DumpFiniteStateInstNodeGraph(node_list, valid_node_set);

  // Step 3: Build FiniteStateNode Graph
  std::unordered_map<size_t, size_t> state_id_trans_map;
  // Step 3.1: Renumber the node IDs and make sparse nodes denser,
  for (size_t id : valid_node_set) {
    state_id_trans_map[id] = state_id_trans_map.size();
    KLLM_LOG_DEBUG << fmt::format("Relate X{} = S{}", state_id_trans_map[id], id);
  }
  for (size_t id : valid_node_set) {
    size_t current_id = state_id_trans_map[id];
    // Step 3.2: Create A Finite State Node
    FiniteStateType state_type = FiniteStateType::NON_GENERATION_STATE;
    if (node_list[id]->edges.empty()) {
      // stop state.
      state_type = FiniteStateType::FINAL_STATE;
    } else if (node_list[id]->edges.size() > 1 || node_list[id]->edges[0].first == "*") {
      // generation state.
      state_type = FiniteStateType::GENERATION_STATE;
    }
    state_map_[current_id] = std::make_shared<FiniteStateNode>(state_type, current_id);

    // Step 3.3: Enumerate all possible transition tokens
    if (state_type == FiniteStateType::GENERATION_STATE) {
      // generation state.
      for (auto& child : node_list[id]->edges) {
        std::vector<int> next_tokens;
        if (child.first == "*") {
          if (node_list[child.second]->edges.empty()) {
            KLLM_LOG_WARNING << fmt::format(
                "Finite State Machine build warning: generation state {} does not have next state.", current_id);
            state_map_[current_id]->edges_map_[kDefaultToken] = std::make_pair("*", current_id);
            continue;
          }
          /* The word '*' represents that the current state can generate any character. To exit this state, the
           * beginning of the next state must be hit. For example, "name": "[*]", When in the generation state of "*",
           * only when the output is ", it will transition to the next state. Therefore, all possible ending states
           * need to be enumerated.*/
          const std::string& next_string = node_list[child.second]->edges[0].first;
          size_t next_state_id = node_list[child.second]->edges[0].second;
          TokenizeWithSpecialSymbols(next_string, next_tokens);

          /* Special case: Due to the introduction of a loop structure, there may be multiple branching scenarios:
           * Current state State-A + '}]' = State-End; Current state State-A + '}, ' = Continue reasoning the next
           * segment However, when merging and splitting similar categories, the graph may grow like this:
           * Sx means the State Node x, (y) means the edges's fixed string is y.
           *       S0
           *       | (*)
           *       S1
           *       | (})
           *       S2
           * (]) /   \ (,)
           *   S3    S4
           * The ideal scenario is that in S0, only one "}" is generated, and then in S2, either a "]" or a "}" is
           * generated, leading to different branches. However, since a token itself may represent multiple characters,
           * there is a possibility that in S0, both "}]" or "}," are generated at once, making it impossible to
           * correctly hit the jump token and causing the structured output to fail. To solve this problem, an auxiliary
           * state is added to store inconsistent tokens:\
           *                S0
           *                | (*)
           *                S1
           *              / |  \
           *            /   |    \
           *    (}])  /     |(})   \ (},)
           *        /       S2       \
           *      S5   (])/   \(,)    S6
           *        \    /      \    /
           *          S3          S4
           */
          for (auto& cchild : node_list[next_state_id]->edges) {
            std::vector<int> step_tokens;
            std::string step_string = fmt::format("{}{}", next_string, cchild.first);
            TokenizeWithSpecialSymbols(step_string, step_tokens);
            if (step_tokens != next_tokens) {
              for (int& token : step_tokens) {
                size_t additional_state_id = state_id_trans_map.size();
                size_t step_state_id = state_id_trans_map[cchild.second];
                state_id_trans_map[additional_state_id] = additional_state_id;
                state_map_[current_id]->edges_map_[token] = std::make_pair("*", additional_state_id);
                state_map_[additional_state_id] =
                    std::make_shared<FiniteStateNode>(FiniteStateType::NON_GENERATION_STATE, additional_state_id);
                state_map_[additional_state_id]->edges_map_[kDefaultToken] = std::make_pair(step_string, step_state_id);
              }
            }
          }
        } else {
          TokenizeWithSpecialSymbols(child.first, next_tokens);
        }
        size_t child_id = state_id_trans_map[child.second];
        for (int& token : next_tokens) {
          state_map_[current_id]->edges_map_[token] = std::make_pair(child.first, child_id);
        }
      }
    } else if (state_type == NON_GENERATION_STATE) {
      std::string edge = node_list[id]->edges[0].first;
      size_t child_id = state_id_trans_map[node_list[id]->edges[0].second];
      state_map_[current_id]->edges_map_[kDefaultToken] = std::make_pair(edge, child_id);
      if (!string_tokens_map_.count(edge)) {
        std::vector<int> empty_token_vec;
        TokenizeString2Tokens(edge, empty_token_vec);
      }
    }
  }
  std::unordered_set<size_t> node_set;
  std::string node_graph_str;
  DumpFiniteStateNodeGraph(0, node_set, node_graph_str);
  KLLM_LOG_INFO << fmt::format("BuildFiniteStateMachine Success! Valid Node Size = {}, Relate Map = \n{}",
                               valid_node_set.size(), node_graph_str);
}

FiniteStateMachine::FiniteStateMachine(std::string& str) { BuildFiniteStateMachine(str); }

void FiniteStateMachine::CheckFSMPopToken(const size_t& state_id, std::vector<int>& input_tokens) {
  if (state_map_.find(state_id) == state_map_.end()) {
    KLLM_LOG_ERROR << fmt::format("Invalid State ID {} when checking whether request need pop the last token.",
                                  state_id);
    return;
  }
  if (input_tokens.empty()) {
    return;
  }
  std::shared_ptr<FiniteStateNode>& state_node = state_map_[state_id];
  int trans_token = input_tokens.back();
  if (state_node->edges_map_.count(trans_token) && state_node->edges_map_[trans_token].first == "*") {
    input_tokens.pop_back();
  }
}

FiniteStateType FiniteStateMachine::GetStateType(const size_t& state_id) {
  if (state_map_.count(state_id)) {
    return state_map_[state_id]->state_type_;
  }
  return FiniteStateType::INVALID_STATE;
}

bool FiniteStateMachine::IsStopState(const size_t& state_id) {
  return !state_map_.count(state_id) || state_map_[state_id]->state_type_ == FiniteStateType::FINAL_STATE;
}

std::shared_ptr<FiniteStateMachine> FiniteStateMachineController::CreateOrGetFSM(std::string& str) {
  if (!fsm_map.count(str)) {
    fsm_map[str] = std::make_shared<FiniteStateMachine>(str);
  }
  return fsm_map[str];
}

}  // namespace ksana_llm
