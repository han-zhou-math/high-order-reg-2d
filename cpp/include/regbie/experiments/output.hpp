// Output locations of the C++ drivers.
//
// The Python drivers write below results/. The C++ drivers write the same
// relative paths below an output root, by default cpp/results, so that they
// never overwrite the saved Python results. Every driver accepts
// --output-root DIR (run from the repository root).
#pragma once
#include <cstring>
#include <filesystem>
#include <string>

namespace regbie {

// Map a Python path "results/<rest>" to "<root>/<rest>".
inline std::filesystem::path output_path(const std::filesystem::path& root, const std::string& python_path) {
  const std::string prefix = "results/";
  std::string rest = python_path;
  if (rest.compare(0, prefix.size(), prefix) == 0) rest = rest.substr(prefix.size());
  return root / rest;
}

// Read --output-root from argv (default cpp/results); other arguments are left to the driver.
inline std::filesystem::path output_root(int argc, char** argv) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], "--output-root") == 0) return argv[i + 1];
  return "cpp/results";
}

}  // namespace regbie
