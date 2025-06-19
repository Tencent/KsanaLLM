/* Copyright 2024 Tencent Inc.  All rights reserved.
 * ==============================================================================*/

#include "ksana_llm/utils/finite_state_machine.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/tokenizer.h"
#include "test.h"

namespace ksana_llm {

class FiniteStateMachineTest : public testing::Test {
 protected:
  void SetUp() override {
    fsm_controller_ = std::make_shared<FiniteStateMachineController>();
    Singleton<Tokenizer>::GetInstance()->InitTokenizer("/model/llama-hf/7B");
  }
  void TearDown() override { Singleton<Tokenizer>::GetInstance()->DestroyTokenizer(); }

  std::shared_ptr<FiniteStateMachineController> fsm_controller_;

  size_t start_state_id_ = 0;
  std::unordered_set<size_t> node_set_;
  std::string fsm_dump_str_;
};

TEST_F(FiniteStateMachineTest, RepeatedCreateFSM) {
  std::string pattern = "{'name': '[*]', 'age': [*], 'work': '[*]'}";
  std::shared_ptr<FiniteStateMachine> x = fsm_controller_->CreateOrGetFSM(pattern);
  std::shared_ptr<FiniteStateMachine> y = fsm_controller_->CreateOrGetFSM(pattern);
  ASSERT_EQ(x, y);
}

TEST_F(FiniteStateMachineTest, BuildLinearFSM) {
  /*
   * The simplest finite state machine. The content to be supplemented by the LLM model and the given fixed text appear
   * at regular intervals.
   *  S0
   *  | ({'name': ')
   *  S1
   *  | (*)
   *  S2
   *  | (', 'age': )
   *  S3
   *  | (*)
   *  S4
   *  | (, 'work': ')
   *  S5
   *  | (*)
   *  S6
   *  | ('})
   *  S7
   */
  std::string pattern = "{'name': '[*]', 'age': [*], 'work': '[*]'}";
  std::shared_ptr<FiniteStateMachine> linear_fsm = fsm_controller_->CreateOrGetFSM(pattern);
  node_set_.clear();
  fsm_dump_str_ = "";
  linear_fsm->DumpFiniteStateNodeGraph(start_state_id_, node_set_, fsm_dump_str_);
  ASSERT_EQ(fsm_dump_str_,
            "stateDiagram-v2\n"
            "    [*] --> S0\n"
            "    S0 --> S1 : {'name'COLON '\n"
            "    S1 --> S2 : *\n"
            "    S2 --> S3 : ', 'age'COLON \n"
            "    S3 --> S4 : *\n"
            "    S4 --> S5 : , 'work'COLON '\n"
            "    S5 --> S6 : *\n"
            "    S6 --> S7 : '}\n"
            "    S7 --> [*] : finish!\n");
}

TEST_F(FiniteStateMachineTest, BuildLoopFSM) {
  /*
   * The output of the LLM model is a variable-length JSON list, where each JSON contains fixed keys: "name" and "age".
   * However, the number of JSONs in the output result varies, requiring the introduction of a loop structure.
   *          S0
   *          | ([{'name': ')
   *          S1
   *          | (*)
   *          S2
   *          | (', 'age': )
   *          S3
   *   (*) /      \
   *      S4        \ (*)
   *      |           \
   *  (}) |             \
   *      |               \
   *      S5               S11
   * (])/  ↑\(,{'name': ')  |
   * S10   |   \            |
   *  |    |      \         |
   *  |    |         \      | (}, {'name': ')
   *  |    |           \    |
   * End   |             \  |
   *       |                S6 <-----------|
   *       |                | (*)          |
   *       | (})            S7             |
   *       |                |(', 'age':)   |
   *       |                S8             |
   *       |           (*) /  \ (*)        | (}, {'name': ')
   *       |             S9    S14         |
   *       |______________|     |__________|
   */
  std::string pattern = "\\[{'name': '[*]', 'age': [*]}(?:, {'name': '[*]', 'age': [*]})*\\]";
  std::shared_ptr<FiniteStateMachine> loop_fsm = fsm_controller_->CreateOrGetFSM(pattern);
  node_set_.clear();
  fsm_dump_str_ = "";
  loop_fsm->DumpFiniteStateNodeGraph(start_state_id_, node_set_, fsm_dump_str_);
  ASSERT_EQ(fsm_dump_str_,
            "stateDiagram-v2\n"
            "    [*] --> S0\n"
            "    S0 --> S1 : [{'name'COLON '\n"
            "    S1 --> S2 : *\n"
            "    S2 --> S3 : ', 'age'COLON \n"
            "    S3 --> S4 : *\n"
            "    S3 --> S11 : *\n"
            "    S4 --> S5 : }\n"
            "    S5 --> S10 : ]\n"
            "    S5 --> S6 : , {'name'COLON '\n"
            "    S10 --> [*] : finish!\n"
            "    S6 --> S7 : *\n"
            "    S7 --> S8 : ', 'age'COLON \n"
            "    S8 --> S9 : *\n"
            "    S8 --> S14 : *\n"
            "    S9 --> S5 : }\n"
            "    S14 --> S6 : }, {'name'COLON '\n"
            "    S11 --> S6 : }, {'name'COLON '\n");
}

TEST_F(FiniteStateMachineTest, BuildSelectFSM) {
  /*
   * The user needs the LLM model to select one output from multiple options.
   *            S0
   *            | (Hi, my )
   *            S1
   *          / |  \
   *        /   |    \
   * (work)|    |      \ (age)
   *       |    |(name) |
   *       \    |      /
   *        \   |    /
   *          \ |  /
   *            S2
   *            | ( is )
   *            S3
   *            | (*)
   *            S4
   *            | (.)
   *            S5
   */
  std::string pattern = "Hi, my (age|name|work) is [*]\\.";
  std::shared_ptr<FiniteStateMachine> select_fsm = fsm_controller_->CreateOrGetFSM(pattern);
  node_set_.clear();
  fsm_dump_str_ = "";
  select_fsm->DumpFiniteStateNodeGraph(start_state_id_, node_set_, fsm_dump_str_);
  ASSERT_EQ(fsm_dump_str_,
            "stateDiagram-v2\n"
            "    [*] --> S0\n"
            "    S0 --> S1 : Hi, my \n"
            "    S1 --> S2 : work\n"
            "    S1 --> S2 : name\n"
            "    S1 --> S2 : age\n"
            "    S2 --> S3 :  is \n"
            "    S3 --> S4 : *\n"
            "    S4 --> S5 : .\n"
            "    S5 --> [*] : finish!\n");
}

TEST_F(FiniteStateMachineTest, BuildNoEndFSM) {
  /*
   * The marker [*] appears at the end of the regex. The model does not have a specific end delimiter.
   * S0
   * | (Hi, )
   * S1 <-----|
   * |        | (*)
   * |________|
   */
  std::string pattern = "Hi, [*]";
  std::shared_ptr<FiniteStateMachine> no_end_fsm = fsm_controller_->CreateOrGetFSM(pattern);
  node_set_.clear();
  fsm_dump_str_ = "";
  no_end_fsm->DumpFiniteStateNodeGraph(start_state_id_, node_set_, fsm_dump_str_);
  ASSERT_EQ(fsm_dump_str_,
            "stateDiagram-v2\n"
            "    [*] --> S0\n"
            "    S0 --> S1 : Hi, \n"
            "    S1 --> S1 : *\n");
}

}  // namespace ksana_llm
