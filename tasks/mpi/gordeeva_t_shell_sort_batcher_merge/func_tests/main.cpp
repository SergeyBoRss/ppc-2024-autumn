#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <vector>

#include "mpi/gordeeva_t_shell_sort_batcher_merge/include/ops_mpi.hpp"

TEST(ShellSortSequential, Correctness) {
  const int size = 500;
  auto input_vec = gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::rand_vec(size, 0, 1000);
  auto expected_vec = input_vec;

  std::sort(expected_vec.begin(), expected_vec.end());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
  taskDataSeq->inputs_count = {size};
  std::vector<int> result_seq(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_seq.data()));
  taskDataSeq->outputs_count = {size};

  gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential testSeq(taskDataSeq);

  ASSERT_TRUE(testSeq.validation());
  testSeq.pre_processing();
  testSeq.run();
  testSeq.post_processing();

  ASSERT_EQ(result_seq, expected_vec);
}

TEST(ShellSortParallel, Correctness) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const int size = 500;
  std::vector<int> input_vec;
  std::vector<int> result_parallel(size);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_vec = gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::rand_vec(size, 0, 1000);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
    taskDataPar->inputs_count = {size};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_parallel.data()));
    taskDataPar->outputs_count = {size};
  }

  gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel testPar(taskDataPar);

  ASSERT_TRUE(testPar.validation());
  testPar.pre_processing();
  testPar.run();
  std::cout << "[ MY0 ]" << world.rank() << std::endl;
  testPar.post_processing();

  if (world.rank() == 0) {
    auto expected_vec = input_vec;
    std::sort(expected_vec.begin(), expected_vec.end());

    ASSERT_EQ(result_parallel, expected_vec);
  }
}
