// Copyright 2023 Nesterov Alexander
#include "mpi/gordeeva_t_shell_sort_batcher_merge/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

void gordeeva_t_shell_sort_batcher_merge_mpi::shellSort(std::vector<int>& arr) {
  size_t arr_length = arr.size();
  for (size_t step = arr_length / 2; step > 0; step /= 2) {
    for (size_t i = step; i < arr_length; i++) {
      size_t j = i;
      while (j >= step && arr[j - step] > arr[j]) {
        std::swap(arr[j], arr[j - step]);
        j -= step;
      }
    }
  }
}

std::vector<int> gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::rand_vec(int size, int down, int upp) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(down, upp);

  std::vector<int> v(size);
  for (auto& number : v) {
    number = dis(gen);
  }
  return v;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::pre_processing() {
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    throw std::runtime_error("Invalid input data in pre_processing.");
  }

  size_t sz = taskData->inputs_count[0];
  auto* input_tmp = reinterpret_cast<int32_t*>(taskData->inputs[0]);
  input_.assign(input_tmp, input_tmp + sz);
  res_.resize(sz);

  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::validation() {
  if (taskData->inputs.empty() || taskData->outputs.empty()) return false;
  if (taskData->inputs_count[0] <= 0) return false;
  if (taskData->outputs_count.size() != 1) return false;
  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::run() {
  shellSort(input_);
  res_ = input_;
  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::post_processing() {
  if (taskData->outputs.empty()) {
    throw std::runtime_error("Invalid output data in post_processing.");
  }

  int* output_matr = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), output_matr);
  return true;
}

void gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::batcher_merge(size_t rank1, size_t rank2,
                                                                                 std::vector<int>& local_input_local) {
  int rank = world.rank();
  std::vector<int> received_data;

  if (rank == rank1) {
    // Отправка данных процессу rank2
    std::cout << "[batcher_merge] Rank " << rank << " sending to Rank " << rank2 << std::endl;
    world.send(rank2, 0, local_input_local);
    // Получение данных от процесса rank2
    std::cout << "[batcher_merge] Rank " << rank << " receiving from Rank " << rank2 << std::endl;
    world.recv(rank2, 0, received_data);
    std::cout << "[batcher_merge] Rank " << rank << " received data from Rank " << rank2 << std::endl;
  } else if (rank == rank2) {
    // Получение данных от процесса rank1
    std::cout << "[batcher_merge] Rank " << rank << " receiving from Rank " << rank1 << std::endl;
    world.recv(rank1, 0, received_data);
    std::cout << "[batcher_merge] Rank " << rank << " received data from Rank " << rank1 << std::endl;
    // Отправка данных процессу rank1
    std::cout << "[batcher_merge] Rank " << rank << " sending to Rank " << rank1 << std::endl;
    world.send(rank1, 0, local_input_local);
    std::cout << "[batcher_merge] Rank " << rank << " sent data to Rank " << rank1 << std::endl;
  }

  // Слияние данных, если полученные данные не пусты
  if (!received_data.empty()) {
    std::vector<int> merged_data(local_input_local.size() + received_data.size());
    std::merge(local_input_local.begin(), local_input_local.end(), received_data.begin(), received_data.end(),
               merged_data.begin());

    if (rank == rank1) {
      // Сохранение первых элементов после слияния
      local_input_local.assign(merged_data.begin(), merged_data.begin() + local_input_local.size());
      std::cout << "[batcher_merge] Rank " << rank << " updated local_input_local after merge with Rank " << rank2
                << std::endl;
    } else if (rank == rank2) {
      // Сохранение последних элементов после слияния
      local_input_local.assign(merged_data.end() - local_input_local.size(), merged_data.end());
      std::cout << "[batcher_merge] Rank " << rank << " updated local_input_local after merge with Rank " << rank1
                << std::endl;
    }
  }
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::pre_processing() {
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
      throw std::runtime_error("Invalid input data in pre_processing.");
    }

    sz_mpi = taskData->inputs_count[0];
    auto* input_tmp = reinterpret_cast<int32_t*>(taskData->inputs[0]);
    input_.assign(input_tmp, input_tmp + sz_mpi);
    res_.resize(sz_mpi);
  }
  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::validation() {
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->outputs.empty()) return false;
    if (taskData->inputs_count[0] <= 0) return false;
    if (taskData->outputs_count.size() != 1) return false;
  }
  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::run() {
  size_t rank = world.rank();
  size_t size = world.size();

  // Отладочное сообщение
  std::cout << "[MY0] Process " << rank << " started run()" << std::endl;

  // 1. Broadcast размер массива sz_mpi
  size_t sz_mpi_local = 0;
  boost::mpi::broadcast(world, sz_mpi, 0);  // sz_mpi уже инициализирован в pre_processing
  sz_mpi_local = sz_mpi;                    // Теперь все процессы знают sz_mpi

  // Отладочное сообщение
  std::cout << "[MY1] Process " << rank << " received sz_mpi: " << sz_mpi_local << std::endl;

  // 2. Вычисление распределения данных
  size_t delta = sz_mpi_local / size;
  size_t ost = sz_mpi_local % size;

  std::vector<int> sz_rk(size, static_cast<int>(delta));

  // Распределение остатка
  for (size_t i = 0; i < ost; ++i) {
    sz_rk[i]++;
  }

  // Отладочное сообщение
  if (rank == 0) {
    std::cout << "[MY2] sz_rk: ";
    for (auto sz : sz_rk) std::cout << sz << " ";
    std::cout << std::endl;
  }

  // 3. Broadcast данные input_
  // Только процесс 0 инициализирует input_, остальные резервируют место
  if (rank != 0) {
    input_.resize(sz_mpi_local);
  }
  boost::mpi::broadcast(world, input_.data(), sz_mpi_local, 0);

  // Отладочное сообщение
  std::cout << "[MY3] Process " << rank << " received input_ data." << std::endl;

  // 4. Scatterv для распределения данных
  std::vector<int> local_input(sz_rk[rank]);
  boost::mpi::scatterv(world, input_, sz_rk, local_input.data(), 0);

  // Отладочное сообщение
  std::cout << "[MY4] Process " << rank << " received local_input of size " << local_input.size() << std::endl;

  // 5. Сортировка локального подмассива
  shellSort(local_input);

  // Отладочное сообщение
  std::cout << "[MY5] Process " << rank << " sorted local_input." << std::endl;

  // 6. Пошаговое слияние подмассивов между процессами
  for (size_t i = 0; i < size; ++i) {
    if (rank % 2 == i % 2 && rank + 1 < size) {
      std::cout << "[MY6] Process " << rank << " merging with " << (rank + 1) << std::endl;
      batcher_merge(rank, rank + 1, local_input);
    } else if (rank % 2 != i % 2 && rank > 0) {
      std::cout << "[MY7] Process " << rank << " merging with " << (rank - 1) << std::endl;
      batcher_merge(rank - 1, rank, local_input);
    }
  }

  // 7. Gatherv для сбора отсортированных данных на процесс 0
  if (rank == 0) {
    res_.resize(sz_mpi_local);
  }
  boost::mpi::gatherv(world, local_input.data(), local_input.size(), res_.data(), sz_rk, 0);

  // Отладочное сообщение
  std::cout << "[MY8] Process " << rank << " completed gatherv." << std::endl;

  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::post_processing() {
  if (world.rank() == 0 && !taskData->outputs.empty()) {
    std::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}
