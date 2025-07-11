/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

#include "ksana_llm/multi_batch_controller/multi_batch_controller.h"
#include "ksana_llm/helpers/environment_test_helper.h"

using namespace ksana_llm;

class MultiBatchControllerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { InitLoguru(); }

  void SetUp() override { Reset(4); }

  void Reset(int max_multi_batch_num) {
    max_pp_multi_batch_num_ = max_multi_batch_num;
    start_thread_cnt_ = 0;
    finish_threads_num_ = 0;
    input_data_.resize(max_pp_multi_batch_num_, 0);
    running_order_.clear();
    step_preface_order_.clear();
    step_epilogue_order_.clear();
    recv_order_.clear();
    threads_.clear();
    multi_batch_controller_ = std::make_unique<MultiBatchController>(max_pp_multi_batch_num_);
  }

  void TearDown() override { multi_batch_controller_.reset(); }

  void RecordRecvOrder(size_t batch_id) {
    std::lock_guard<std::mutex> lock(recv_order_mutex_);
    recv_order_.push_back(batch_id);
    std::cout << "Recved Batch ID " << batch_id << std::endl;
  }

  void RecordRunningOrder(size_t batch_id) {
    std::lock_guard<std::mutex> lock(running_order_mutex_);
    running_order_.push_back(batch_id);
    // std::cout << "Running Batch ID " << batch_id << std::endl;
  }

  void RecordStepPrefaceOrder(size_t batch_id) {
    std::lock_guard<std::mutex> lock(step_preface_order_mutex_);
    step_preface_order_.push_back(batch_id);
    std::cout << "Preface Batch ID " << batch_id << std::endl;
  }

  void RecordStepEpilogueOrder(size_t batch_id) {
    std::lock_guard<std::mutex> lock(step_epilogue_order_mutex_);
    step_epilogue_order_.push_back(batch_id);
    std::cout << "Epilogue Batch ID " << batch_id << std::endl;
  }

  void ProcessData(size_t batch_id) {
    std::this_thread::sleep_for(std::chrono::milliseconds((batch_id * 23) % 8));
  }

  void StepPreface(size_t batch_id) {
    std::this_thread::sleep_for(std::chrono::milliseconds((batch_id * 23) % 50));
  }

  void StepEpilogue(size_t batch_id) {
    std::this_thread::sleep_for(std::chrono::milliseconds((batch_id * 23) % 10));
  }

  void StartThreads(bool need_sync_start) {
    need_sync_start_ = need_sync_start;
    threads_.clear();
    for (size_t batch_id = 0; batch_id < max_pp_multi_batch_num_; ++batch_id) {
      threads_.emplace_back(&MultiBatchControllerTest::ThreadFunc, this, batch_id);
    }
  }

  void JoinThreads() {
    for (auto& thread : threads_) {
      thread.join();
    }
  }

  void WaitAllThreadsFinish() {
    std::unique_lock<std::mutex> lock(finish_mutex_);
    finish_cv_.wait(lock, [this] { return finish_threads_num_ == max_pp_multi_batch_num_; });
  }

  void ThreadFunc(size_t batch_id) {
    int run_data_cnt = 0;
    while (true) {
      // std::cout << "Thread " << batch_id << " run data id " << run_data_cnt << std::endl;
      ProcessData(batch_id);
      if (run_data_cnt >= input_data_[batch_id]) {
        multi_batch_controller_->NotifyCurrentBatchThreadNotReady(batch_id);
        break;
      }

      multi_batch_controller_->MultiBatchThreadIsReady(batch_id);
      if (need_sync_start_ && run_data_cnt == 0) {
        std::unique_lock<std::mutex> lock(start_mutex_);
        start_thread_cnt_++;
        start_cv_.notify_all();
        start_cv_.wait(lock, [this] { return start_thread_cnt_ == max_pp_multi_batch_num_; });
      }

      multi_batch_controller_->WaitUtilCurrentBatchCanRun(batch_id);
      RecordRunningOrder(batch_id);
      RecordStepPrefaceOrder(batch_id);
      StepPreface(batch_id);

      multi_batch_controller_->NotifyLastBatchHiddenUnitCanRecv(batch_id);
      multi_batch_controller_->NotifyOtherBatchCanRun();

      multi_batch_controller_->WaitUtilCanRecvCurrentHiddenUnits(batch_id);
      RecordRecvOrder(batch_id);

      multi_batch_controller_->WaitUtilCurrentBatchCanRun(batch_id);
      RecordRunningOrder(batch_id);
      RecordStepEpilogueOrder(batch_id);
      StepEpilogue(batch_id);

      run_data_cnt++;
    }
    {
      std::lock_guard<std::mutex> lock(finish_mutex_);
      ++finish_threads_num_;
      finish_cv_.notify_one();
    }
    // std::cout << "Thread " << batch_id << " exited." << std::endl;
  }

 protected:
  // batches cycles
  std::vector<int> input_data_;

  std::mutex start_mutex_;
  std::condition_variable start_cv_;
  bool need_sync_start_ = false;
  int start_thread_cnt_ = 0;

  std::mutex finish_mutex_;
  std::condition_variable finish_cv_;
  int finish_threads_num_ = 0;

  std::vector<std::thread> threads_;

  std::mutex running_order_mutex_;
  std::mutex step_preface_order_mutex_;
  std::mutex step_epilogue_order_mutex_;
  std::mutex recv_order_mutex_;
  std::vector<size_t> running_order_;
  std::vector<size_t> step_preface_order_;
  std::vector<size_t> step_epilogue_order_;
  std::vector<size_t> recv_order_;
  std::unique_ptr<MultiBatchController> multi_batch_controller_ = nullptr;
  int max_pp_multi_batch_num_;
};

// 测试多线程环境下running id的有序执行
TEST_F(MultiBatchControllerTest, BlalancedRunningOrderTest) {
  int max_multi_batch_num = 4;
  Reset(max_multi_batch_num);

  // prepare input data
  int data_num_each_thread = 5;
  input_data_.resize(max_multi_batch_num);
  for (int i = 0; i < max_multi_batch_num; ++i) {
    input_data_[i] = data_num_each_thread;
  }

  bool need_sync_start = true;
  StartThreads(need_sync_start);

  WaitAllThreadsFinish();

  EXPECT_EQ(recv_order_.size(), data_num_each_thread * max_multi_batch_num);
  EXPECT_EQ(step_preface_order_.size(), data_num_each_thread * max_multi_batch_num);
  EXPECT_EQ(step_epilogue_order_.size(), data_num_each_thread * max_multi_batch_num);
  EXPECT_EQ(running_order_.size(), 2 /* preface and epilogue */ * data_num_each_thread * max_multi_batch_num);
  // recv/step preface/step epilogue order: 0,1,2,3,0,1,2,3
  // running order: 0,1,2,3,0,0,1,1,2,2,3,3,0,1,2,3
  for (size_t data_id = 0; data_id < data_num_each_thread; ++data_id) {
    for (size_t batch_id = 0; batch_id < max_multi_batch_num; ++batch_id) {
      size_t idx = data_id * max_multi_batch_num + batch_id;
      EXPECT_EQ(recv_order_.at(idx), batch_id) << "batch_id: " << batch_id << ", data_id: " << data_id;
      EXPECT_EQ(step_preface_order_.at(idx), batch_id) << "batch_id: " << batch_id << ", data_id: " << data_id;
      EXPECT_EQ(step_epilogue_order_.at(idx), batch_id) << "batch_id: " << batch_id << ", data_id: " << data_id;
    }
  }
  for (size_t batch_id = 0; batch_id < max_multi_batch_num ; ++batch_id) {
    size_t st = 0;
    for (size_t data_id = 0; data_id < data_num_each_thread; ++data_id) {
      size_t step_preface_pos = st + (data_id == 0 ? batch_id : (batch_id * 2 + 1));
      size_t step_epilogue_pos =
          st + (data_id == 0 ? 1 : 2) * max_multi_batch_num + batch_id * (data_id == data_num_each_thread - 1 ? 1 : 2);
      EXPECT_EQ(running_order_.at(step_preface_pos), batch_id) << "batch_id: " << batch_id << ", data_id: " << data_id;
      EXPECT_EQ(running_order_.at(step_epilogue_pos), batch_id) << "batch_id: " << batch_id << ", data_id: " << data_id;
      st += max_multi_batch_num * (data_id == 0 ? 1 : 2);
    }
  }
  JoinThreads();
}

TEST_F(MultiBatchControllerTest, DoNotSyncStart) {
  int max_multi_batch_num = 8;
  Reset(max_multi_batch_num);

  // prepare input data
  int data_num_each_thread = 5;
  input_data_.resize(max_multi_batch_num);
  for (int i = 0; i < max_multi_batch_num; ++i) {
    input_data_[i] = data_num_each_thread;
  }

  bool need_sync_start = false;
  StartThreads(need_sync_start);

  WaitAllThreadsFinish();

  EXPECT_EQ(recv_order_.size(), data_num_each_thread * max_multi_batch_num);
  EXPECT_EQ(step_preface_order_.size(), data_num_each_thread * max_multi_batch_num);
  EXPECT_EQ(step_epilogue_order_.size(), data_num_each_thread * max_multi_batch_num);
  EXPECT_EQ(running_order_.size(), 2 * data_num_each_thread * max_multi_batch_num);

  for (size_t i = 0; i < data_num_each_thread * max_multi_batch_num; ++i) {
    EXPECT_EQ(step_preface_order_.at(i), recv_order_.at(i));
    EXPECT_EQ(step_epilogue_order_.at(i), recv_order_.at(i));
  }

  JoinThreads();
}

// 测试多线程环境下输入不平衡的情况
TEST_F(MultiBatchControllerTest, UnBlalancedNotSyncStartOrderTest) {
  int max_multi_batch_num = 2;
  Reset(max_multi_batch_num);

  // prepare input data
  input_data_.resize(2);
  input_data_ = {3, 6};

  bool need_sync_start = false;
  StartThreads(need_sync_start);

  WaitAllThreadsFinish();

  EXPECT_EQ(recv_order_.size(), 9);
  EXPECT_EQ(step_preface_order_.size(), 9);
  EXPECT_EQ(step_epilogue_order_.size(), 9);
  EXPECT_EQ(running_order_.size(), 18);

  for (size_t i = 0; i < 9; ++i) {
    EXPECT_EQ(step_preface_order_.at(i), recv_order_.at(i)) << i;
    EXPECT_EQ(step_epilogue_order_.at(i), recv_order_.at(i)) << i;
  }

  JoinThreads();
}

// 测试多线程环境下输入不平衡的情况
TEST_F(MultiBatchControllerTest, UnBlalancedSyncStartOrderTest) {
  int max_multi_batch_num = 2;
  Reset(max_multi_batch_num);

  // prepare input data
  input_data_.resize(2);
  input_data_ = {8, 1};

  bool need_sync_start = true;
  StartThreads(need_sync_start);

  WaitAllThreadsFinish();

  EXPECT_EQ(recv_order_.size(), 9);
  EXPECT_EQ(step_preface_order_.size(), 9);
  EXPECT_EQ(step_epilogue_order_.size(), 9);
  EXPECT_EQ(running_order_.size(), 18);

  for (size_t i = 0; i < 9; ++i) {
    EXPECT_EQ(step_preface_order_.at(i), recv_order_.at(i)) << i;
    EXPECT_EQ(step_epilogue_order_.at(i), recv_order_.at(i)) << i;
  }

  JoinThreads();
}
