// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/plekhanov_d_allreduce_mine/include/ops_mpi.hpp"

TEST(plekhanov_d_allreduce_mine_perf_test, test_pipeline_run) {
  int rows = 5000;
  int columns = 5000;
  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> max_vec_mpi(columns, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = std::vector<int>(rows * columns, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_vec_mpi.data()));
    taskDataPar->outputs_count.emplace_back(max_vec_mpi.size());
  }

  auto testMpiTaskParallel = std::make_shared<plekhanov_d_allreduce_mine_mpi::TestMPITaskBoostParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (unsigned i = 0; i < max_vec_mpi.size(); i++) {
      EXPECT_EQ(0, max_vec_mpi[0]);
    }
  }
}

TEST(plekhanov_d_allreduce_mine_perf_test, test_task_run) {
  int rows = 5000;
  int columns = 5000;
  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> max_vec_mpi(columns, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = std::vector<int>(rows * columns, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_vec_mpi.data()));
    taskDataPar->outputs_count.emplace_back(max_vec_mpi.size());
  }

  auto testMpiTaskParallel = std::make_shared<plekhanov_d_allreduce_mine_mpi::TestMPITaskBoostParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (unsigned i = 0; i < max_vec_mpi.size(); i++) {
      EXPECT_EQ(0, max_vec_mpi[0]);
    }
  }
}