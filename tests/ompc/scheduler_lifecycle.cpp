// Regression test for OMPC scheduler initialization, graph dumping and cleanup.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

extern "C" {
void __kmpc_InitScheduler();
void __kmpc_DeinitScheduler();
void __kmpc_dumpTaskGraph();
bool __kmpc_is_legacy_scheduler();
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr,
                 "Usage: %s <roundrobin|graph_roundrobin|heft_legacy>\n",
                 argv[0]);
    return 1;
  }

  const std::string scheduler = argv[1];
  const bool legacy = scheduler == "heft_legacy";

  if (!legacy && scheduler != "roundrobin" &&
      scheduler != "graph_roundrobin") {
    std::fprintf(stderr, "Unsupported scheduler: %s\n", argv[1]);
    return 1;
  }

  const char *configured = std::getenv("OMPCLUSTER_SCHEDULER");
  if (!configured || scheduler != configured) {
    std::fprintf(stderr, "OMPCLUSTER_SCHEDULER mismatch\n");
    return 1;
  }

  const char *prefix = std::getenv("OMPCLUSTER_TASK_GRAPH_DUMP_PATH");
  if (!legacy && !prefix) {
    std::fprintf(stderr, "Missing OMPCLUSTER_TASK_GRAPH_DUMP_PATH\n");
    return 1;
  }

  __kmpc_InitScheduler();

  if (__kmpc_is_legacy_scheduler() != legacy) {
    std::fprintf(stderr, "Unexpected scheduler implementation\n");
    __kmpc_DeinitScheduler();
    return 2;
  }

  if (!legacy)
    __kmpc_dumpTaskGraph();

  __kmpc_DeinitScheduler();

  if (!legacy) {
    for (int index = 0; index < 2; ++index) {
      const std::string path =
          std::string(prefix) + "_graph_" + std::to_string(index) + ".dot";

      std::ifstream file(path);
      if (!file) {
        std::fprintf(stderr, "Missing DOT file: %s\n", path.c_str());
        return 3;
      }

      const std::string content(
          (std::istreambuf_iterator<char>(file)),
          std::istreambuf_iterator<char>());

      if (content.find("digraph TargetTaskGraph") == std::string::npos) {
        std::fprintf(stderr, "Invalid DOT file: %s\n", path.c_str());
        return 4;
      }
    }
  }

  std::printf("PASS: scheduler lifecycle (%s)\n", argv[1]);
  return 0;
}
