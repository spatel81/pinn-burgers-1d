// ═══════════════════════════════════════════════════════════════════════
// PINN Burgers 1D — C++ Inference Driver
// ═══════════════════════════════════════════════════════════════════════
//
// Single executable that orchestrates the full PINN workflow:
//
//   [1] Train & export (Python)  →  calls export/export_model.py via
//       system() to train the PINN and save a TorchScript model.
//
//   [2] Load (LibTorch)  →  loads the .pt file with torch::jit::load(),
//       moves it to the target device (CPU or XPU).
//
//   [3] Infer (LibTorch)  →  evaluates u(x, t) on a grid and writes
//       results to CSV.
//
// Usage:
//   ./pinn_inference --device xpu --output-dir build/output
//   ./pinn_inference --skip-training --model-path model.pt --device cpu
//   ./pinn_inference --device cpu --epochs 1000 --no-lbfgs
//
// Target platform: Aurora supercomputer (Intel PVC GPUs, icpx compiler)
// Build: see CMakeLists.txt and README.md
//

#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


// ═══════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════

struct Config {
    std::string device       = "cpu";
    std::string output_dir   = "build/output";
    std::string model_path   = "";       // Empty = auto (output_dir + /pinn_burgers_traced.pt)
    int         epochs       = 15000;
    bool        no_lbfgs     = false;
    bool        skip_training = false;
    int         nx           = 256;      // Spatial grid points for inference
    int         nt           = 5;        // Time slices for inference
};


// ═══════════════════════════════════════════════════════════════════════
// ARGUMENT PARSING
// ═══════════════════════════════════════════════════════════════════════

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "\n"
        << "Train a PINN (via Python) and run inference (via LibTorch).\n"
        << "\n"
        << "Options:\n"
        << "  --device <dev>       Inference device: cpu or xpu (default: cpu)\n"
        << "  --output-dir <dir>   Output directory (default: build/output)\n"
        << "  --model-path <path>  Path to existing TorchScript model (skips training)\n"
        << "  --epochs <n>         Adam training epochs (default: 15000)\n"
        << "  --no-lbfgs           Skip L-BFGS fine-tuning\n"
        << "  --skip-training      Skip training, use existing model\n"
        << "  --nx <n>             Spatial grid points for inference (default: 256)\n"
        << "  --nt <n>             Time slices for inference (default: 5)\n"
        << "  --help               Show this message\n";
}

Config parse_args(int argc, char* argv[]) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--device" && i + 1 < argc) {
            cfg.device = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            cfg.output_dir = argv[++i];
        } else if (arg == "--model-path" && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            cfg.epochs = std::stoi(argv[++i]);
        } else if (arg == "--no-lbfgs") {
            cfg.no_lbfgs = true;
        } else if (arg == "--skip-training") {
            cfg.skip_training = true;
        } else if (arg == "--nx" && i + 1 < argc) {
            cfg.nx = std::stoi(argv[++i]);
        } else if (arg == "--nt" && i + 1 < argc) {
            cfg.nt = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    return cfg;
}


// ═══════════════════════════════════════════════════════════════════════
// STEP 1: TRAIN AND EXPORT (via Python)
// ═══════════════════════════════════════════════════════════════════════
//
// Invokes the Python export script via system().  This is the simplest
// and most robust approach on Aurora, where the Python environment is
// managed by `module load frameworks` and embedding the interpreter
// would add fragile build dependencies.

void train_and_export(const Config& cfg) {
    std::ostringstream cmd;
    cmd << "python export/export_model.py"
        << " --device " << cfg.device
        << " --output-dir " << cfg.output_dir
        << " --epochs " << cfg.epochs;
    if (cfg.no_lbfgs) {
        cmd << " --no-lbfgs";
    }

    std::cout << "\n"
              << "=================================================================\n"
              << "  STEP 1: TRAINING AND EXPORTING MODEL (Python)\n"
              << "=================================================================\n"
              << "  Command: " << cmd.str() << "\n\n";

    int ret = std::system(cmd.str().c_str());
    if (ret != 0) {
        std::cerr << "\n  ERROR: Python training failed (exit code "
                  << ret << ")\n";
        std::exit(1);
    }
}


// ═══════════════════════════════════════════════════════════════════════
// STEP 2: LOAD TORCHSCRIPT MODEL
// ═══════════════════════════════════════════════════════════════════════

torch::jit::script::Module load_model(const std::string& model_path,
                                       torch::Device device) {
    std::cout << "\n"
              << "=================================================================\n"
              << "  STEP 2: LOADING TORCHSCRIPT MODEL\n"
              << "=================================================================\n"
              << "  Model file: " << model_path << "\n";

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        std::cerr << "  ERROR: Failed to load model: " << e.what() << "\n";
        std::exit(1);
    }

    model.to(device);
    model.eval();

    std::cout << "  Model loaded and moved to " << device << "\n";
    return model;
}


// ═══════════════════════════════════════════════════════════════════════
// STEP 3: RUN INFERENCE
// ═══════════════════════════════════════════════════════════════════════
//
// Evaluate the trained model on a regular grid:
//   x ∈ [−1, 1]  (nx points)
//   t ∈ [0, 1]   (nt evenly spaced slices)
//
// Output is written to inference_results.csv with columns: x, t, u_pred

void run_inference(torch::jit::script::Module& model,
                   torch::Device device,
                   const Config& cfg) {
    // Domain bounds (must match the training domain)
    const float x_min = -1.0f, x_max = 1.0f;
    const float t_min =  0.0f, t_max = 1.0f;

    // Build spatial and temporal grids
    auto x_vals = torch::linspace(x_min, x_max, cfg.nx,
                                   torch::dtype(torch::kFloat32));
    auto t_vals = torch::linspace(t_min, t_max, cfg.nt,
                                   torch::dtype(torch::kFloat32));

    // Open output CSV
    std::string csv_path = cfg.output_dir + "/inference_results.csv";
    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        std::cerr << "  ERROR: Cannot open " << csv_path << " for writing\n";
        std::exit(1);
    }
    csv << "x,t,u_pred\n";

    std::cout << "\n"
              << "=================================================================\n"
              << "  STEP 3: RUNNING INFERENCE\n"
              << "=================================================================\n"
              << "  Grid: " << cfg.nx << " x-points × "
              << cfg.nt << " t-slices\n"
              << "  Device: " << device << "\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    // Disable gradient computation for inference
    torch::NoGradGuard no_grad;

    for (int j = 0; j < cfg.nt; ++j) {
        float t_val = t_vals[j].item<float>();

        // Build (nx, 2) input tensor: column 0 = x, column 1 = t
        auto x_col = x_vals.unsqueeze(1);                          // (nx, 1)
        auto t_col = torch::full({cfg.nx, 1}, t_val,
                                  torch::dtype(torch::kFloat32));  // (nx, 1)
        auto input = torch::cat({x_col, t_col}, /*dim=*/1)
                         .to(device);                              // (nx, 2)

        // Forward pass through the TorchScript model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        auto output = model.forward(inputs).toTensor();            // (nx, 1)

        // Move result to CPU for writing
        auto u_pred = output.to(torch::kCPU).contiguous();
        auto u_data = u_pred.accessor<float, 2>();

        // Write this time slice to CSV
        for (int i = 0; i < cfg.nx; ++i) {
            csv << x_vals[i].item<float>() << ","
                << t_val << ","
                << u_data[i][0] << "\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(
                            t_end - t_start).count();

    csv.close();

    int total_points = cfg.nx * cfg.nt;
    std::cout << "  Inference complete: " << elapsed_ms << " ms"
              << " (" << total_points << " points)\n"
              << "  Results saved: " << csv_path << "\n";
}


// ═══════════════════════════════════════════════════════════════════════
// DEVICE SELECTION
// ═══════════════════════════════════════════════════════════════════════

torch::Device select_device(const std::string& requested) {
    if (requested == "xpu") {
        try {
            torch::Device dev("xpu");
            // Test with a small allocation to verify XPU is functional
            auto test = torch::zeros({1}, torch::TensorOptions().device(dev));
            std::cout << "  Device: xpu (Intel GPU)\n";
            return dev;
        } catch (const c10::Error& e) {
            std::cerr << "  WARNING: XPU requested but not available ("
                      << e.what() << ")\n"
                      << "  Falling back to CPU.\n";
            return torch::Device(torch::kCPU);
        }
    }

    std::cout << "  Device: cpu\n";
    return torch::Device(torch::kCPU);
}


// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    std::cout << "\n"
              << "=================================================================\n"
              << "  PINN Burgers 1D — C++ Inference Pipeline\n"
              << "=================================================================\n";

    // Create output directory
    std::string mkdir_cmd = "mkdir -p " + cfg.output_dir;
    std::system(mkdir_cmd.c_str());

    // Determine model path
    std::string model_path = cfg.model_path;
    if (model_path.empty()) {
        model_path = cfg.output_dir + "/pinn_burgers_traced.pt";
    }

    // ── Step 1: Train and export (unless skipped) ───────────────────
    if (!cfg.skip_training && cfg.model_path.empty()) {
        train_and_export(cfg);
    } else {
        std::cout << "\n  Training skipped — using existing model: "
                  << model_path << "\n";
    }

    // ── Step 2: Select device and load model ────────────────────────
    torch::Device device = select_device(cfg.device);
    auto model = load_model(model_path, device);

    // ── Step 3: Run inference ───────────────────────────────────────
    run_inference(model, device, cfg);

    // ── Summary ─────────────────────────────────────────────────────
    std::cout << "\n"
              << "=================================================================\n"
              << "  DONE\n"
              << "=================================================================\n"
              << "  Model:   " << model_path << "\n"
              << "  Output:  " << cfg.output_dir << "/inference_results.csv\n"
              << "\n";

    return 0;
}
