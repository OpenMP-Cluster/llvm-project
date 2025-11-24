//===---------------------- kmp_target_heft.cpp - HEFT scheduler
//---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Heterogeneous Earliest Finish Time scheduling algorithm.
//
//===----------------------------------------------------------------------===//

#include "kmp_target_heft.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <set>
#include <deque>
#include <array>
#include <utility>
#include <cmath>

namespace detail {

// Implementation of HEFTContext
// =============================================================================

HEFTContext::HEFTContext(ScheduleMap &Schedule, kmp_tdg_info *tdg)
    : CommMultiplier(1.0), DefaultCompCost(100.0), DefaultCommCost(1.0),
      NumThreads(1), tdg(tdg),
      Schedule(Schedule) {

  if (tdg != nullptr)
    computeUpwardRanks();
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_COMM_COEF"))
    CommMultiplier = std::stod(EnvStr);
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_COMP_COST"))
    DefaultCompCost = std::stod(EnvStr);
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_COMM_COST"))
    DefaultCommCost = std::stod(EnvStr);
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_LOG"))
    HEFTLogPath = EnvStr;

  if (char *EnvStr = std::getenv("OMPCLUSTER_NUM_EXEC_EVENT_HANDLERS"))
    NumThreads = std::stoi(EnvStr);

}

void HEFTContext::computeUpwardRanks() {

  KG_TRACE(10,("[HEFT] Computing upward rank for all tasks.\n"));

  assert(tdg != nullptr && "The task map should not be NULL");
  for (auto *Task : ReverseTopologicalOrder(tdg)) {
    double MaxSuccUrankComm = 0.0;
    for (int32_t idx_succ = 0 ; idx_succ < Task->nsuccessors ; idx_succ++) {
      auto Succ = &(tdg->record_map[Task->successors[idx_succ]]);
      MaxSuccUrankComm =
          std::max(MaxSuccUrankComm,
                   UpwardRank[Succ] + computeAvgCommCost(*Task, *Succ));
    }
    UpwardRank[Task] = computeAvgCompCost(*Task) + MaxSuccUrankComm;
  }
}

double HEFTContext::computeEST(TaskPtr Task, DeviceId Device) {
  double MaxPredEST = 0.0;
  // Iterate over predecessors
  for (int32_t idx_pred = 0 ; idx_pred < Task->npredecessors ; idx_pred++) {

    // Ignore deferred tasks for now
    if (isDeferredTask(*Task))
      continue;
//    if(Schedule.count(KMP_TDG_TASKID_TO_TASKID(tdg, Task->predecessors[idx_pred])) != 1)
//	printf("Task %d, predecessor %d\n",KMP_TASKPTR_TO_TDG_TASKID(Task),Task->predecessors[idx_pred]);
    assert(Schedule.count(KMP_TDG_TASKID_TO_TASKID(tdg, Task->predecessors[idx_pred])) == 1 &&
           "Predecessor should already be scheduled!");

    DeviceId DevicePred(Schedule[KMP_TDG_TASKID_TO_TASKID(tdg,Task->predecessors[idx_pred])]);
    auto Pred = &(tdg->record_map[Task->predecessors[idx_pred]]);
    auto Candidate =
        AFT[Pred] + computeCommCost(*Pred, DevicePred, *Task, Device);

    if (Candidate > MaxPredEST)
      MaxPredEST = Candidate;
  }

  // Find the earliest start time if task is scheduled on device.
  EST[Task][Device] = std::max(DeviceAvail[Device], MaxPredEST);
  return EST[Task][Device];
}

double HEFTContext::computeEFT(TaskPtr Task, DeviceId Device) {
  // Calculate EST for task id, rank pair if it does not exist
  if (EST.count(Task) == 0 || EST[Task].count(Device) == 0)
    computeEST(Task, Device);

  EFT[Task][Device] = computeCompCost(*Task, Device) + EST[Task][Device];
  return EFT[Task][Device];
}

double HEFTContext::scheduleTask(TaskPtr Task, DeviceId Device) {

  Schedule[KMP_TASKPTR_TO_TASKID(Task)] = {
      .state = TaskState::SCHEDULED,
      .rank = Device.Rank,
      .thread = Device.Thread,
      .translated_rank = -1,
      .task = Task,
  };

  assert(Schedule[KMP_TASKPTR_TO_TASKID(Task)].state == TaskState::SCHEDULED);
  AST[Task] = EST[Task][Device];
  AFT[Task] = EFT[Task][Device];
  return AFT[Task];
}

double HEFTContext::scheduleTaskOnBestRank(TaskPtr Task) {
  const auto &EFTs = EFT.at(Task);

  auto BestRank = std::min_element(
      EFTs.begin(), EFTs.end(),
      [](const auto &L, const auto &R) { return L.second < R.second; });

  DeviceAvail[BestRank->first] = scheduleTask(Task, BestRank->first);
  return DeviceAvail[BestRank->first];
}

void HEFTContext::forceScheduleTask(TaskPtr Task, DeviceId Device) {
  computeEFT(Task, Device);
  scheduleTaskOnBestRank(Task);
}

void HEFTContext::deferTask(TaskPtr Task) {
  // Do not change state of FAILED tasks during the reschedule
  if (Schedule[KMP_TASKPTR_TO_TASKID(Task)].state != TaskState::FAILED)
    Schedule[KMP_TASKPTR_TO_TASKID(Task)].state = TaskState::DEFERRED;
}

PriorityQueue HEFTContext::buildQueue() const {
  PriorityQueue Pqueue{UpwardRank.begin(), UpwardRank.end()};

  std::sort(Pqueue.begin(), Pqueue.end(),
            [](const auto &L, const auto &R) { return L.second > R.second; });

  return Pqueue;
}

std::vector<DeviceId> HEFTContext::buildDeviceList(int32_t NumRanks) const {
  std::vector<DeviceId> Devices;
  Devices.reserve(NumRanks);
  for (int32_t R = 1; R <= NumRanks; R++)
    for (int32_t H = 0; H < NumThreads; H++)
      Devices.emplace_back(R, H);
  return Devices;
}

void HEFTContext::clear() {
  EST.clear();
  EFT.clear();
  AST.clear();
  AFT.clear();
  UpwardRank.clear();
  DeviceAvail.clear();
  CommunicationCost.clear();
  AverageComputationCost.clear();
  ComputationCost.clear();
}

void HEFTContext::clearCompCost() {
  AverageComputationCost.clear();
  ComputationCost.clear();
}

void HEFTContext::setTaskMap(int32_t tdg_id) {
  tdg = __kmpc_find_tdg(tdg_id);
  assert(tdg != nullptr && "Expected a non-null task map!");
}

double HEFTContext::computeCommCost(const kmp_node_info &A,
                                    DeviceId DeviceA,
                                    const kmp_node_info &B,
                                    DeviceId DeviceB) {
  // If the tasks are to execute on the same rank, we do not pay for the
  // communication because no data needs to be moved.
  if (DeviceA.Rank == DeviceB.Rank)
    return 0.0;

  // FIXME: take actual bandwidth between RankA and RankB into consideration
  // instead of just using the average cost.
  return computeAvgCommCost(A, B);
}

/// Search for all `out` dependencies of A which are also `in` dependencies of B
/// and accumulate the size of those dependencies (in bytes).
double HEFTContext::computeAvgCommCost(const kmp_node_info &A,
                                       const kmp_node_info &B) {

  double TotalBytes = 0.0;
  // If the communication cost was already computed, then use the cached value.
  auto ItA = CommunicationCost.find(&A);
  if (ItA != CommunicationCost.end()) {
    auto ItB = ItA->second.find(&B);
    if (ItB != ItA->second.end()) {
      return ItB->second;
    }
  }
  
  // Look for arguments which are common between tasks A and B and accumulate
  // their sizes in bytes.
  if(A.target_data !=nullptr && B.target_data!=nullptr){
    for (int32_t I = 0; I < A.target_data->num_args; I++) {
      for (int32_t J = 0; J < B.target_data->num_args; J++) {
        if (A.target_data->args[I] == B.target_data->args[J]) {
          TotalBytes += A.target_data->arg_sizes[I];
        }
      }
    }
  }
  CommunicationCost[&A][&B] = TotalBytes;

  KG_TRACE(10,("[HEFT] Dependency `%d -> %d` has average cost: %lf KB\n", KMP_TASKNODE_TO_TASKID(A),
     KMP_TASKNODE_TO_TASKID(B), TotalBytes / 1024.0));

  return CommMultiplier * TotalBytes;
}

double HEFTContext::computeAvgCompCost(const kmp_node_info &Task) {
  KG_TRACE(10,("[HEFT] computeAvgCompCost (start) : tdg task id: %d\n", KMP_TASKNODE_TO_TASKID(Task)));
  // If computation cost was already computed, then use the cached value.
  auto It = AverageComputationCost.find(&Task);
  if (It != AverageComputationCost.end()) {
    KG_TRACE(10,("[HEFT] computeAvgCompCost (end) : Using cached value\n"));
    return It->second;
  }

  //TODO Something went wrong, we should always have a value after extrapolation.
  KMP_DEBUG_ASSERT(false);
  KG_TRACE(10,("[HEFT] computeAvgCompCost (end) : Using default value\n"));
    return AverageComputationCost[&Task] = 100.0;
}


double HEFTContext::computeCompCost(const kmp_node_info &Task,
                                    DeviceId Device) {
  
  KG_TRACE(10,("[HEFT] computeCompCost (start) : tdg task id: %d, device id %d\n", KMP_TASKNODE_TO_TASKID(Task), Device.Rank-1));
  //If it exists, we use the recorded time from the tdg structure

  auto It = ComputationCost.find(&Task);
  if (It != ComputationCost.end()) {
    auto Value = It->second.find(Device.Rank-1);
    if (Value != It->second.end()) {
      return Value->second;
    }
  }
  
  //If we don't have the task time for the precise device, use the average time
  //Something went wrong, we should always have a value after extrapolation.
  KMP_DEBUG_ASSERT(false);
  KG_TRACE(10,("[HEFT] computeCompCost : Using Averaged Comp cost"));
    return ComputationCost[&Task][Device.Rank-1] = computeAvgCompCost(Task);
}

void HEFTContext::computeAvgCompCosts(){
  //TODO should we construct it once then pass it as an argument?
  for (auto Taskptr : TopologicalOrder(tdg)){
    double accumulated_time = 0.0;
    kmp_int32 profiled_device = 0;
    for(auto DeviceValue : ComputationCost[Taskptr]) {
      accumulated_time += DeviceValue.second;
      profiled_device++;
    } 
    AverageComputationCost[Taskptr] = accumulated_time/profiled_device;
  }
}

void HEFTContext::homogeneousInterpolation() {
  computeAvgCompCosts();
  for (auto Taskptr : TopologicalOrder(tdg)){
    double average_value = AverageComputationCost[Taskptr];
    for( kmp_int32 device = 0; device < __kmpc_get_num_devices(); device++ ) {
      if(!ComputationCost[Taskptr].count(device))
        ComputationCost[Taskptr][device] = average_value;
    }
  }
}

void HEFTContext::generalProportionalInterpolation() {

  kmp_int32 device_number = __kmpc_get_num_devices();
  std::vector<std::vector<double>> beta(device_number,std::vector<double>(device_number,-1.0)); // We want to compute the betas (proportional coefficient between the devices' speeds)

//Computation of the Betas

 for( kmp_int32 device_0 = 0; device_0 < device_number ; device_0++ ) {
  for ( kmp_int32 device_1 = device_0 ; device_1 < device_number ; device_1++){
    double sum = 0.0;
    kmp_int32 size = 0; 
    for (auto Taskptr : TopologicalOrder(tdg)){
      if(isTargetDataTask(Taskptr) || !ComputationCost[Taskptr].count(device_0) ||  !ComputationCost[Taskptr].count(device_1))
        continue; 
      sum += std::log(ComputationCost[Taskptr][device_1]/ComputationCost[Taskptr][device_0]);  
      size++;
     }
     if(size) {
       beta[device_0][device_1] = std::exp(sum/size) ; 
       beta[device_1][device_0] = 1/beta[device_0][device_1];   
       KG_TRACE(10,("[HEFT] Getting profile for beta %d, %d: %d\n",device_0, device_1));
     }
     else {
       KG_TRACE(10,("[HEFT] No profile for beta %d, %d\n",device_0, device_1));
     }
   }
 }
//TODO print the initial result

//Interpolation of the pivot Beta[0][X]

  double sum_b0 = 0.0;
  kmp_int32 size_b0 = 0; 
  for(auto beta0 : beta[0] ){
    if( beta0 > 0.0) {
      sum_b0 += std::log(beta0);
      size_b0++;
    } 
  } 
  
  double geomean_b0 = std::exp(sum_b0/size_b0);
  
//  std::deque<int> devices_to_visite;
//  auto  it = devices_to_visite.begin();
//  for (int i = 0; i < device_number ; ++i) {
//    devices_to_visite.push_back(i);
//  }
//  kmp_int32 b0_count = 0;
//
//  while(b0_count < device_number) {
//    while (!devices_to_visite.empty()) { 
//      kmp_int32 device_1 = devices_to_visite.front();
//      devices_to_visite.pop_front();
//      if(beta[0][device_1] == -1.0) // We don't have the value yet, we can't search another beta from it
//        continue; 
//      b0_count++;
//      for ( kmp_int32 device_2 = 1 ; device_2 < device_number ; device_2++) {
//        if ( beta[0][device_2] != -1.0 ) //We already know this beta
//          continue;
//        if ( beta[device_1][device_2] == -1.0) //We need this beta to compute the new one
//          continue;
//        beta[0][device_2] = beta[0][device_1] * beta[device_1][device_2]; 
//        beta[device_2][0] = 1/beta[0][device_2]; 
//        b0_count++;
//        devices_to_visite.push_back(device_2);
//     }
//     if (b0_count == device_number)
//       break;
//   }
//   if (b0_count < device_number) { //We add a value with the partial homogeneous hypothesis
//     for(kmp_int32 device_1 = 1 ; device_1 <= device_number ; device_1++ ){
//       if (beta[0][device_1] <= 0.0 ){
//         beta[0][device_1] = geomean_b0;
//         beta[device_1][0] = 1/geomean_b0 ;
//         b0_count++; 
//         devices_to_visite.push_back(device_1);
//         break;
//       } 
//     } 
//   }
//  }
  
  std::deque<kmp_int32> devices_to_visite;
  for (kmp_int32 i = 0; i < device_number ; ++i) {
    devices_to_visite.push_back(i);
  }
  while(!devices_to_visite.empty()) {
    kmp_int32 queue_size = devices_to_visite.size();
    for (kmp_int32 it = 0 ; it < queue_size; ++it){
      kmp_int32 device_1 = devices_to_visite.front();
      devices_to_visite.pop_front();
      if(beta[0][device_1] != -1.0) {
      // We already have the value
        continue;
      }
      double sum =0.0;
      kmp_int32 size = 0;
      for (kmp_int32 device_2 = 1 ; device_2 < device_number; device_2++) {
        if (beta[0][device_2] != -1 && beta[device_2][device_1] != -1) {
          sum += std::log(beta[0][device_2]*beta[device_2][device_1]);
          size++;
        }
      } 
      if (size){
        beta[0][device_1] = std::exp(sum/size); 
        beta[device_1][0] =  1/beta[0][device_1];
      }
      else {
        devices_to_visite.push_back(device_1); //We were not able to determine a value for B0,d1, we will have to go back to it
      }      
    }
    if(devices_to_visite.size() == queue_size ) { //We need to do a partial homogenous assumptions to continue computing the pivot     
       beta[0][devices_to_visite.front()] = geomean_b0;   
       beta[devices_to_visite.front()][0] =  1/beta[0][devices_to_visite.front()];
       devices_to_visite.pop_front();
    }
  }
//TODO print the computed pivot

//Interpolation of the Betas[X][Y]
  for( kmp_int32 device_0 = 1; device_0 < device_number ; device_0++ ) {
    std::vector<kmp_int32> device_list;
    //Searching for a suitable Beta for beta_0
    for ( kmp_int32 device_1 = device_0 + 1 ; device_1 < device_number ; device_1++){
      if(beta[device_0][device_1] == -1.0) // We don't have the value yet, we can't search another beta from it
        continue;
      device_list.push_back(device_1);
    }
   
    for ( kmp_int32 device_1 = device_0 + 1 ; device_1 < device_number ; device_1++){
      if(beta[device_0][device_1] != -1.0) // We already have a value, no need to complete
        continue;
      double sum = 0.0;
      for (kmp_int32 second_device : device_list){
        sum += std::log(beta[device_0][second_device]*beta[second_device][0]*beta[0][device_1]);
      }
      beta[device_0][device_1] =  std::exp(sum/device_list.size());
      beta[device_1][device_0] =  1/beta[device_0][device_1];
    }
  } 
//  for( kmp_int32 device_0 = 1; device_0 < device_number ; device_0++ ) {
//    kmp_int32 second_device;
//    //Searching for a suitable Beta for beta_0
//    for ( kmp_int32 device_1 = device_0 + 1 ; device_1 < device_number ; device_1++){
//      if(beta[device_0][device_1] == -1.0) // We don't have the value yet, we can't search another beta from it
//        continue;
//      second_device = device_1;
//      break;
//    }
//   
//    for ( kmp_int32 device_1 = device_0 + 1 ; device_1 < device_number ; device_1++){
//      if(beta[device_0][device_1] != -1.0) // We already have a value, no need to complete
//        continue;
//      beta[device_0][device_1] =  beta[device_0][second_device]*beta[second_device][0]*beta[0][device_1];
//      beta[device_1][device_0] =  1/beta[device_0][device_1];
//    }
//  }
//TODO print the resulting beta

// We just need to compute the costs now
  
  for (auto Taskptr : TopologicalOrder(tdg)){
    if(isTargetDataTask(Taskptr))
      continue; 
    //Copy known profiled costs for the task
    std::unordered_map<kmp_int32, double> profiled_comp_cost;  
    kmp_int32 size= ComputationCost[Taskptr].size();    
    for(auto CC: ComputationCost[Taskptr]) { 
      profiled_comp_cost[CC.first] = CC.second; 
      KMP_DEBUG_ASSERT(CC.second > 0.0); 
    }
    for( kmp_int32 device_0 = 0; device_0 < device_number ; device_0++ ) {
      //We find a pair (task,device) that was not profiled
      if(ComputationCost[Taskptr].count(device_0))
        continue;
      //We take the average of all the path from the tasks that were profiled 
      double sum  = 0.0;
      for(auto [profiled_device, profiled_value]: profiled_comp_cost) {
        sum += std::log(profiled_value* beta[profiled_device][device_0]);
      }
      ComputationCost[Taskptr][device_0]= std::exp(sum/size);
    }
  }
}

void HEFTContext::extrapolateCompCost(kmp_int32 scheduling_iteration_counter, kmp_int32 scheduling_policy, kmp_int32 initial_iteration) {

  KG_TRACE(10,("[HEFT] extrapolateCompCost (start)\n"));
  //__kmpc_print_profiled_comp_cost(tdg);

  bool is_incomplete_profile = false; 
  for (auto Taskptr : TopologicalOrder(tdg)){
    for( kmp_int32 device = 0; device < __kmpc_get_num_devices(); device++ ) {
      if(isTargetDataTask(*Taskptr)) { 
        ComputationCost[Taskptr][device]= 1.0;
      }
      else {
        KG_TRACE(10,("[Inter HEFT]: Task %d, device %d : iteration number: %d\n",KMP_TASKPTR_TO_TASKID(Taskptr),device,tdg->record_map[KMP_TASKPTR_TO_TDG_TASKID(Taskptr)].task_time[device].iteration_number));
        if(tdg->record_map[KMP_TASKPTR_TO_TDG_TASKID(Taskptr)].task_time[device].iteration_number) {
          KG_TRACE(10,("[Inter HEFT]: Adding task %d , device %d to CompCost\n",KMP_TASKPTR_TO_TDG_TASKID(Taskptr),device));
          ComputationCost[Taskptr][device]=static_cast<double>(tdg->record_map[KMP_TASKPTR_TO_TDG_TASKID(Taskptr)].task_time[device].mean_time);
        }
        else { 
         KG_TRACE(10,("[Inter HEFT]: Task %d , device %d : no profile\n",KMP_TASKPTR_TO_TDG_TASKID(Taskptr),device));
          is_incomplete_profile = true;
          if(!scheduling_iteration_counter) //No profile information : set the task value to 100.0
            ComputationCost[Taskptr][device]= 100.0;
        }
      }
    }
  }
  
 //We could simplify these ifs with a switch into computeAvgCompCosts()

  if(!is_incomplete_profile || !scheduling_iteration_counter) {
    computeAvgCompCosts();
    KG_TRACE(10,("[HEFT] extrapolateCompCost (end): no need for extrapolation\n "));
    return;
  }
  
  if (scheduling_iteration_counter == 1) {
    homogeneousInterpolation();
    computeAvgCompCosts();
    KG_TRACE(10,("[HEFT] extrapolateCompCost (end):homogeneous hypothesis\n  "));
    return;
  }

  //general case
  //TODO: implement it ( in case we can't have a sufficent condition to compute the Interpolation with partial homogeneous hypothessis)  
  generalProportionalInterpolation(); 
  computeAvgCompCosts();

  KG_TRACE(10,("[HEFT] extrapolateCompCost (end): general case\n"));

//TODO:
//If Less organized : normal matrix completion (proportional hypothesis) compute ACC[T]

}



/// Dump internal HEFT information to a file.
void HEFTContext::dump(int32_t NumDevices) {
  if (HEFTLogPath.empty())
    return;

  FILE *fp = fopen(HEFTLogPath.c_str(), "w");

  if (fp == NULL) {
    KG_TRACE(10,("[HEFT] Failed to open log file.\n"));
    return;
  }

  auto Devices = buildDeviceList(NumDevices);

  KG_TRACE(10,("[HEFT] Dumping HEFT stats to file '%s'\n", HEFTLogPath.c_str()));

  // FIXME: Print the header with correct columns for EST/EFT
  fprintf(fp,
          "%-4s  %-10s  %-10s  %-10s  %-10s  %-10s  %-7s  %-7s  %-7s  %-7s"
          "%-34s  %-34s\n",
          "ID", "FNID", "Rank", "AST", "AFT", "Dev", "Root", "Exit", "Data",
          "Classic", "EST", "EFT");

  std::vector<kmp_node_info *> TasksV;
  TasksV.reserve(tdg->num_tasks);
  for (int i = 0; i < tdg->num_tasks; i++) {
    TasksV.push_back(&(tdg->record_map[i]));
  }

  std::sort(TasksV.begin(), TasksV.end(),
            [&](kmp_node_info *A, kmp_node_info *B) {
              return UpwardRank[A] > UpwardRank[B];
            });

  // Iterate over tasks
  for (auto const *task_ptr : TasksV) {
    auto &task = *task_ptr;
    uint64_t ofid = (uint64_t)task.target_data->outlined_fn_id;
    fprintf(fp,
            "%04d  0x%08lx  %10.2lf  %10.2lf  %10.2lf  %7d  %7d  %7d  %7d  %4d",
            KMP_TASKPTR_TO_TASKID(task_ptr), ofid, UpwardRank[&task], AST[&task], AFT[&task],
            Schedule[KMP_TASKPTR_TO_TASKID(task_ptr)].rank, isRootTask(task), isExitTask(task),
            isTargetDataTask(task), isClassicalTask(task));
    for (const auto &Device : Devices)
      fprintf(fp, "  %10.2lf", EST[&task][Device]);
    for (const auto &Device : Devices)
      fprintf(fp, "  %10.2lf", EFT[&task][Device]);
    fprintf(fp, "\n");
  }

  fprintf(fp, "\n");
  fprintf(fp, "Dependencies\n");
  fprintf(fp, "-------------------------------------------------------\n");

  for (const auto &A : CommunicationCost) {
    auto id_A = KMP_TASKPTR_TO_TASKID(A.first);
    for (const auto &B : A.second) {
      auto id_B = KMP_TASKPTR_TO_TASKID(B.first);
      fprintf(fp, "%04d -> %04d : %10.4lf\n", id_A, id_B, B.second);
    }
  }

  fprintf(fp, "\n");
  fprintf(fp, "Schedule Map\n");
  fprintf(fp, "-------------------------------------------------------\n");

  for (const auto &s : Schedule) {
    fprintf(fp, "%04d: %04d %d %d\n", s.first, KMP_TASKPTR_TO_TASKID(s.second.task),
            s.second.rank, s.second.state);
  }

  fclose(fp);
}

} // namespace detail

// Implementation of HEFTScheduler
// =============================================================================

HEFTScheduler::DumpOptions::DumpOptions() {
  if (char *Value = std::getenv("OMPCLUSTER_HEFT_DUMP_EDGE_LABEL"))
    EdgeLabel = atoi(Value) > 0;
  if (char *Value = std::getenv("OMPCLUSTER_HEFT_DUMP_PRED"))
    Predecessors = atoi(Value) > 0;
  if (char *Value = std::getenv("OMPCLUSTER_HEFT_DUMP_FAKE_ENTRY"))
    FakeEntryNode = atoi(Value) > 0;
  if (char *Value = std::getenv("OMPCLUSTER_HEFT_DUMP_FAKE_EXIT"))
    FakeExitNode = atoi(Value) > 0;
}

/// This is the high-level implementation of the HEFT algorithm as is described
/// in the original paper.
///
/// \see https://ieeexplore.ieee.org/document/993206
void HEFTScheduler::scheduleGraph() {
  using detail::DeviceId;
  acquireNewGraphPost();
  auto PQueue = Ctx.buildQueue();
  auto NumDevices = __kmpc_get_num_devices();
  auto Devices = Ctx.buildDeviceList(NumDevices);

  KG_TRACE(10,("[SCHEDULER] Scheduling graph with HEFT algorithm:\n"));

  while (not PQueue.empty()) {
    const auto *Task = PQueue.front().first;
    PQueue.pop_front();

    // Do not reschedule already executed or saved tasks
    const auto Data = schedule_map[KMP_TASKPTR_TO_TASKID(Task)];
    if (Data.state > TaskState::SCHEDULED && Data.state != TaskState::FAILED)
      continue;

    // Defer the scheduling for target data tasks
    if (isTargetDataTask(Task)) {
      Ctx.deferTask(Task);
      continue;
    }

    // Classical tasks are pinned in the head node (rank=0)
    if (isClassicalTask(Task)) {
      Ctx.forceScheduleTask(Task, DeviceId::head());
      continue;
    }

    // If the user specifies a device for a task, it must be executed there
    if (hasSpecifiedDevice(Task) != 0) {
      Ctx.forceScheduleTask(Task, DeviceId(Task->target_data->device_id+1));
      continue;
    }

    for (const auto &Device : Devices)
      Ctx.computeEFT(Task, Device);

    Ctx.scheduleTaskOnBestRank(Task);
  }
  // Schedule deferred tasks
  for (auto &Entry : schedule_map) {
    auto &Data = Entry.second;
    const auto *Task = Data.task;

    // Do not reschedule already executed or saved tasks
    if (Data.state > TaskState::SCHEDULED && Data.state != TaskState::FAILED)
      continue;

    // These are already scheduled.
    if (not Ctx.isDeferredTask(*Task))
      continue;

    KG_TRACE(10,("[HEFT] Scheduling deferred task %d: targetdata=%d root=%d exit=%d "
       "nsuccs=%d npreds=%d\n",
       KMP_TASKPTR_TO_TASKID(Task), isTargetDataTask(Task), isRootTask(Task),
       isExitTask(Task), Task->nsuccessors, Task->npredecessors));

    double MaxCost = 0.0;
    int32_t Rank = 1;

    // Find the successor with highest communication cost
    // NOTE: Exit target data tasks are temporary scheduled but the rank will
    // be overwritten later with PinTask function.

    for (int32_t idx_succ = 0 ; idx_succ < Task->nsuccessors ; idx_succ++) {
      auto Succ = &(tdg->record_map[Task->successors[idx_succ]]);
      auto SuccRank = schedule_map[KMP_TASKPTR_TO_TASKID(Succ)].rank;
      // Save successor ranks of each scheduled data task to detect broadcast
      // opportunities
      dataTaskSuccessorRanks[KMP_TASKPTR_TO_TASKID(Task)].insert(SuccRank);

      if (Ctx.CommunicationCost[Task][Succ] > MaxCost &&
          !isClassicalTask(Succ) && SuccRank != -1) {
        MaxCost = Ctx.CommunicationCost[Task][Succ];
        Rank = SuccRank;
      }
    }

    // Schedule this task to the same rank as the successor chosen before
    Ctx.forceScheduleTask(Task, DeviceId(Rank));
  }

  auto Predicate = [](const auto &Entry) {
    if (Entry.second.state >= TaskState::DISPATCHED)
      return true; // skip checking for executed/saved tasks
    return Entry.second.state == TaskState::SCHEDULED &&
                   isClassicalTask(Entry.second.task)
               ? Entry.second.rank == 0
               : Entry.second.rank > 0;
  };

  // Tasks can be DISPATCHED, SCHEDULED, EXECUTED or FAILED (because of
  // reschedule)

  assert(std::all_of(schedule_map.begin(), schedule_map.end(), Predicate) &&
         "Some tasks were not properly scheduled.");

  addMemoryTasks();

  Ctx.dump(NumDevices);
  KG_TRACE(10,("[SCHEDULER] End HEFT schedule.\n"));
}

/// Reset context and recompute upward ranks for the new graph
void HEFTScheduler::acquireNewGraphPost() {
  Ctx.clear();
  Ctx.setTaskMap(__kmpc_curr_tdg_idx());
  Ctx.extrapolateCompCost(scheduling_iteration_counter, initial_schedule_policy, initial_iteration);
  Ctx.computeUpwardRanks();
}

void HEFTScheduler::dumpTaskGraph() {
  if (graph_dump_dir_path.empty())
    return;

  if (tdg == nullptr) {
    KG_TRACE(10,("[HEFT] Task map is null, skipping dump.\n"));
    return;
  }

  std::ofstream dot(graph_dump_dir_path + "_graph_" +
                    std::to_string(graph_counter) + ".dot");
  // Print header
  dot << "// Generated in HEFTScheduler\n"
      << "digraph TargetTaskGraph {\n"
      << "  labelloc=\"t\";\n"
      << "  label=\"HEFT Scheduler\";\n";

  // For each task
  for (int32_t tid = 0; tid < tdg->num_tasks; tid++) {
    auto task = &(tdg->record_map[tid]);
    int32_t trank = schedule_map[KMP_TDG_TASKID_TO_TASKID(tdg, tid)].rank;
    std::string tinfo = schedule_map[KMP_TDG_TASKID_TO_TASKID(tdg, tid)].source_location;
    int32_t state = static_cast<int32_t>(schedule_map[KMP_TDG_TASKID_TO_TASKID(tdg,tid)].state);

    // Print graph node
    dot << "  V" << tid << " [label=<<B>ID " << tid
        << "</B><BR /><FONT POINT-SIZE=\"10\">Rank " << trank
        << "<BR /> State: " << state;

    if (!tinfo.empty()) {
      dot << "<BR />" << tinfo;
    }

    dot << "</FONT>>";

    if (isTargetDataTask(task))
      dot << " color=gray70 fontcolor=gray50";
    if (isRootTask(task) || isExitTask(task))
      dot << " shape=octagon";

    dot << "];\n";

    // Print forward arrows using the list of successors
    for (int32_t idx = 0; idx < task->nsuccessors; idx++) {
      int32_t sid = task->successors[idx];
      auto succ = &(tdg->record_map[task->successors[idx]]);
      int32_t srank = schedule_map[KMP_TDG_TASKID_TO_TASKID(tdg,sid)].rank;
      const auto *color = detail::getEdgeColor(trank, srank);
      double CommCost = Ctx.CommunicationCost[task][succ];

      // Print graph edge
      dot << "  V" << tid << " -> V" << sid << " [color=\"" << color << "\"";
      if (Dump.EdgeLabel)
        dot << " label=<" << CommCost << ">";
      dot << "];\n";
    }

    // Print backward arrows using the list of predecessors
    if (Dump.Predecessors) {
      for (int32_t pid = 0; pid < task->npredecessors; pid++) {
        dot << "  V" << pid << " -> V" << tid
            << " [style=dashed dir=back color=gray30];\n";
      }
    }
  }

  // Print a single fake entry node
  if (Dump.FakeEntryNode) {
    dot << "  Entry;\n";
    for (int32_t rid = 0; rid < tdg->num_roots; rid++) {
      dot << "  Entry -> V" << rid << ";\n";
    }
  }

    // Print a single fake exit node
    if (Dump.FakeExitNode) {
      dot << "  Exit;\n";
      for (int32_t tid = 0; tid < tdg->num_tasks; tid++) {
        auto task = &(tdg->record_map[tid]);
        if (isExitTask(task))
          dot << "  V" << tid << " -> Exit;\n";
      }
    }
    dot << "}\n";
}

// fixme Return the list of successor ranks of a given data task
std::set<int> HEFTScheduler::getDataTaskSuccessors(int32_t task_id) {
  return dataTaskSuccessorRanks[task_id];
}

void HEFTScheduler::resetWeight(){
  Ctx.clearCompCost();
}

