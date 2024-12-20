#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <vector>

#include "mpi/gordeeva_t_shell_sort_batcher_merge/include/ops_mpi.hpp"

// TEST(gordeeva_t_shell_sort_batcher_merge_mpi, Shell_sort_Zero_Value) {
//   boost::mpi::environment env;
//   boost::mpi::communicator world;
//
//   const int size = 0;
//   std::vector<int> input_vec;
//   std::vector<int> result_parallel(size);
//
//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//   if (world.rank() == 0) {
//     input_vec = gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::rand_vec(size, 0, 1000);
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
//     taskDataPar->inputs_count = {size};
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_parallel.data()));
//     taskDataPar->outputs_count = {size};
//   }
//
//   gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel testPar(taskDataPar);
//
//   ASSERT_FALSE(testPar.validation());
// }

TEST(gordeeva_t_shell_sort_batcher_merge_mpi, Shell_sort_500_with_random) {
  // Инициализация MPI среды
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const int size = 500;
  std::vector<int> input_vec;
  std::vector<int> result_parallel(size);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Генерация случайного вектора на процессе 0
    input_vec = gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::rand_vec(size, 0, 1000);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
    taskDataPar->inputs_count = {static_cast<size_t>(size)};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_parallel.data()));
    taskDataPar->outputs_count = {static_cast<size_t>(size)};
  }

  gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel testPar(taskDataPar);

  // Проверка валидности данных
  ASSERT_TRUE(testPar.validation()) << "Validation failed for Shell_sort_500_with_random";

  // Выполнение этапов задачи
  ASSERT_TRUE(testPar.pre_processing()) << "Pre-processing failed for Shell_sort_500_with_random";
  ASSERT_TRUE(testPar.run()) << "Run failed for Shell_sort_500_with_random";
  ASSERT_TRUE(testPar.post_processing()) << "Post-processing failed for Shell_sort_500_with_random";

  // Синхронизация процессов
  world.barrier();

  if (world.rank() == 0) {
    // Сортировка ожидания на процессе 0 для проверки
    auto expected_vec = input_vec;
    std::sort(expected_vec.begin(), expected_vec.end());

    // Сравнение результатов
    ASSERT_EQ(result_parallel, expected_vec) << "Sorted result does not match expected for size 500";
  }
}

TEST(gordeeva_t_shell_sort_batcher_merge_mpi, Shell_sort_1000_with_random) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const int size = 1000;
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
  std::cout << "[ MY-1 ]" << world.rank() << std::endl;
  testPar.run();
  std::cout << "[ MY1 ]" << world.rank() << std::endl;
  testPar.post_processing();

  world.barrier();

  if (world.rank() == 0) {
    auto expected_vec = input_vec;
    std::sort(expected_vec.begin(), expected_vec.end());

    ASSERT_EQ(result_parallel, expected_vec);
  }
}
