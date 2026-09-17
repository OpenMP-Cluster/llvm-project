#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <iomanip>
#include <cstdint>

struct Model
{
    uintptr_t model_id;
    int worker_id;

    int n;

    double sx;
    double sy;
    double sxx;
    double sxy;

    double alpha;
    double beta;
};

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr
            << "Usage: ompc_perfmodel_display <perf_models.dat>\n";
        return 1;
    }

    const char *path = argv[1];

    std::ifstream file(path);

    if (!file.is_open()) {
        std::cerr
            << "Cannot open performance model file: "
            << path << "\n";
        return 1;
    }

    std::map<uintptr_t, std::map<int, Model>> models;

    std::string line;

    while (std::getline(file, line)) {

        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);

        Model model{};

        iss
            >> std::hex >> model.model_id
            >> std::dec >> model.worker_id
            >> model.n
            >> model.sx
            >> model.sy
            >> model.sxx
            >> model.sxy
            >> model.alpha
            >> model.beta;

        if (iss.fail())
            continue;

        models[model.model_id][model.worker_id] = model;
    }

    std::cout
        << "OMPC DMDAS Performance Models\n"
        << "=============================\n\n";

    for (const auto &[model_id, workers] : models) {

        std::cout
            << "Kernel model: 0x"
            << std::hex << model_id
            << std::dec << "\n\n";

        for (const auto &[worker_id, model] : workers) {

            double mean =
                model.n > 0
                    ? model.sy / model.n
                    : 0.0;

            std::cout
                << "  Worker " << worker_id << "\n"
                << "    samples : " << model.n << "\n"
                << "    mean    : "
                << std::setprecision(8)
                << mean << " s\n"
                << "    alpha   : "
                << model.alpha << "\n"
                << "    beta    : "
                << model.beta << "\n"
                << "    model   : T(x) = "
                << model.alpha
                << " + "
                << model.beta
                << " * x\n\n";
        }
    }

    return 0;
}
