// Output in the formats the Python drivers write: CSV via csv.DictWriter
// (CRLF line ends, floats as repr) and JSON via json.dumps(indent=2).
#pragma once
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace regbie {

// Python repr of a float: shortest round-trip digits, e.g. 0.1, 1e-05, 100.0, nan, inf.
std::string py_repr(double value);

// Ordered JSON value (keys keep insertion order, as Python dicts).
class Json {
 public:
  using Array = std::vector<Json>;
  using Object = std::vector<std::pair<std::string, Json>>;

  Json() : value_(nullptr) {}
  Json(std::nullptr_t) : value_(nullptr) {}
  Json(bool v) : value_(v) {}
  Json(int v) : value_(static_cast<std::int64_t>(v)) {}
  Json(long v) : value_(static_cast<std::int64_t>(v)) {}
  Json(long long v) : value_(static_cast<std::int64_t>(v)) {}
  Json(unsigned long v) : value_(static_cast<std::int64_t>(v)) {}
  Json(double v) : value_(v) {}
  Json(const char* v) : value_(std::string(v)) {}
  Json(std::string v) : value_(std::move(v)) {}
  Json(Array v) : value_(std::make_shared<Array>(std::move(v))) {}
  Json(Object v) : value_(std::make_shared<Object>(std::move(v))) {}
  Json(const std::vector<double>& v);
  Json(const std::vector<int>& v);

  static Json array() { return Json(Array{}); }
  static Json object() { return Json(Object{}); }

  // Object access: inserts the key if missing (keeps first-insertion order).
  Json& operator[](const std::string& key);
  const Json& at(const std::string& key) const;
  bool contains(const std::string& key) const;
  void push_back(Json v);

  bool is_object() const;
  bool is_array() const;
  bool is_number() const;
  double number() const;
  const Array& items() const;
  const Object& fields() const;

  // json.dumps(value, indent=indent, sort_keys=sort_keys); indent < 0 gives the compact
  // single-line form json.dumps(value).
  std::string dump(int indent = 2, bool sort_keys = false) const;

  // Minimal reader for the result files written by the Python code or by dump().
  static Json parse(const std::string& text);

 private:
  void dump_to(std::string& out, int indent, int level, bool sort_keys) const;
  std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, std::shared_ptr<Array>,
               std::shared_ptr<Object>>
      value_;
};

// Write json.dumps(value, indent=2, sort_keys) + "\n", creating parent directories.
void write_json(const std::filesystem::path& path, const Json& value, bool sort_keys = false);
Json read_json(const std::filesystem::path& path);

// A CSV cell as csv.DictWriter writes it: str(value).
using CsvCell = std::variant<std::string, double, std::int64_t, bool>;
std::string csv_text(const CsvCell& cell);
using CsvRow = std::map<std::string, CsvCell>;

// csv.DictWriter(..., fieldnames=fields): header then rows, "\r\n" line ends.
// A field missing from a row is written as an empty cell (restval="").
void write_csv(const std::filesystem::path& path, const std::vector<std::string>& fields,
               const std::vector<CsvRow>& rows);

// Read a CSV written by csv.DictWriter: header and rows as strings.
struct CsvTable {
  std::vector<std::string> fields;
  std::vector<std::map<std::string, std::string>> rows;
};
CsvTable read_csv(const std::filesystem::path& path);

// UTC timestamp as datetime.now(timezone.utc).isoformat().
std::string utc_timestamp();

}  // namespace regbie
