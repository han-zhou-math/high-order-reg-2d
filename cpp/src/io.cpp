#include "regbie/io.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace regbie {

std::string py_repr(double value) {
  if (std::isnan(value)) return "nan";
  if (std::isinf(value)) return value > 0 ? "inf" : "-inf";
  if (value == 0) return std::signbit(value) ? "-0.0" : "0.0";
  // Shortest round-trip digits in scientific form, e.g. "1.2345e-05".
  char buffer[64];
  auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific);
  std::string sci(buffer, result.ptr);
  const auto epos = sci.find('e');
  std::string mantissa = sci.substr(0, epos);
  const int exponent = std::stoi(sci.substr(epos + 1));
  bool negative = false;
  if (mantissa[0] == '-') {
    negative = true;
    mantissa.erase(0, 1);
  }
  std::string digits;
  for (char c : mantissa)
    if (c != '.') digits += c;
  std::string out;
  // Python repr: scientific when the exponent is < -4 or >= 16.
  if (exponent < -4 || exponent >= 16) {
    out = digits.substr(0, 1);
    if (digits.size() > 1) out += "." + digits.substr(1);
    char exp[16];
    std::snprintf(exp, sizeof(exp), "e%c%02d", exponent < 0 ? '-' : '+', std::abs(exponent));
    out += exp;
  } else if (exponent < 0) {
    out = "0." + std::string(-exponent - 1, '0') + digits;
  } else {
    const int int_digits = exponent + 1;
    if (static_cast<int>(digits.size()) <= int_digits) {
      out = digits + std::string(int_digits - digits.size(), '0') + ".0";
    } else {
      out = digits.substr(0, int_digits) + "." + digits.substr(int_digits);
    }
  }
  return negative ? "-" + out : out;
}

Json::Json(const std::vector<double>& v) : value_(std::make_shared<Array>()) {
  auto& a = *std::get<std::shared_ptr<Array>>(value_);
  for (double x : v) a.emplace_back(x);
}

Json::Json(const std::vector<int>& v) : value_(std::make_shared<Array>()) {
  auto& a = *std::get<std::shared_ptr<Array>>(value_);
  for (int x : v) a.emplace_back(x);
}

Json& Json::operator[](const std::string& key) {
  if (std::holds_alternative<std::nullptr_t>(value_)) value_ = std::make_shared<Object>();
  auto& object = *std::get<std::shared_ptr<Object>>(value_);
  for (auto& kv : object)
    if (kv.first == key) return kv.second;
  object.emplace_back(key, Json());
  return object.back().second;
}

const Json& Json::at(const std::string& key) const {
  for (const auto& kv : fields())
    if (kv.first == key) return kv.second;
  throw std::out_of_range("missing JSON key: " + key);
}

bool Json::contains(const std::string& key) const {
  if (!is_object()) return false;
  for (const auto& kv : fields())
    if (kv.first == key) return true;
  return false;
}

void Json::push_back(Json v) {
  if (std::holds_alternative<std::nullptr_t>(value_)) value_ = std::make_shared<Array>();
  std::get<std::shared_ptr<Array>>(value_)->push_back(std::move(v));
}

bool Json::is_object() const { return std::holds_alternative<std::shared_ptr<Object>>(value_); }
bool Json::is_array() const { return std::holds_alternative<std::shared_ptr<Array>>(value_); }
bool Json::is_number() const {
  return std::holds_alternative<double>(value_) || std::holds_alternative<std::int64_t>(value_);
}
double Json::number() const {
  if (std::holds_alternative<double>(value_)) return std::get<double>(value_);
  if (std::holds_alternative<std::int64_t>(value_)) return static_cast<double>(std::get<std::int64_t>(value_));
  throw std::runtime_error("JSON value is not a number");
}
const Json::Array& Json::items() const { return *std::get<std::shared_ptr<Array>>(value_); }
const Json::Object& Json::fields() const { return *std::get<std::shared_ptr<Object>>(value_); }

namespace {
std::string quote(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20 || c >= 0x80) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out + "\"";
}

std::string json_float(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  return py_repr(v);
}
}  // namespace

void Json::dump_to(std::string& out, int indent, int level, bool sort_keys) const {
  const bool pretty = indent >= 0;
  auto newline = [&](int lvl) {
    if (pretty) out += "\n" + std::string(static_cast<size_t>(indent) * lvl, ' ');
  };
  if (std::holds_alternative<std::nullptr_t>(value_)) out += "null";
  else if (std::holds_alternative<bool>(value_)) out += std::get<bool>(value_) ? "true" : "false";
  else if (std::holds_alternative<std::int64_t>(value_)) out += std::to_string(std::get<std::int64_t>(value_));
  else if (std::holds_alternative<double>(value_)) out += json_float(std::get<double>(value_));
  else if (std::holds_alternative<std::string>(value_)) out += quote(std::get<std::string>(value_));
  else if (is_array()) {
    const auto& a = items();
    if (a.empty()) { out += "[]"; return; }
    out += "[";
    for (size_t i = 0; i < a.size(); ++i) {
      if (i) out += pretty ? "," : ", ";
      newline(level + 1);
      a[i].dump_to(out, indent, level + 1, sort_keys);
    }
    newline(level);
    out += "]";
  } else {
    Object o = fields();
    if (o.empty()) { out += "{}"; return; }
    if (sort_keys)
      std::stable_sort(o.begin(), o.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    out += "{";
    for (size_t i = 0; i < o.size(); ++i) {
      if (i) out += pretty ? "," : ", ";
      newline(level + 1);
      out += quote(o[i].first) + ": ";
      o[i].second.dump_to(out, indent, level + 1, sort_keys);
    }
    newline(level);
    out += "}";
  }
}

std::string Json::dump(int indent, bool sort_keys) const {
  std::string out;
  dump_to(out, indent, 0, sort_keys);
  return out;
}

namespace {
struct Parser {
  const std::string& s;
  size_t i = 0;
  void ws() { while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; }
  bool starts(const char* w) { return s.compare(i, std::char_traits<char>::length(w), w) == 0; }
  Json value() {
    ws();
    if (i >= s.size()) throw std::runtime_error("unexpected end of JSON");
    const char c = s[i];
    if (c == '{') {
      ++i;
      Json obj = Json::object();
      ws();
      if (s[i] == '}') { ++i; return obj; }
      while (true) {
        ws();
        const std::string key = string();
        ws();
        if (s[i++] != ':') throw std::runtime_error("expected ':' in JSON");
        obj[key] = value();
        ws();
        if (s[i] == ',') { ++i; continue; }
        if (s[i] == '}') { ++i; return obj; }
        throw std::runtime_error("expected ',' or '}' in JSON");
      }
    }
    if (c == '[') {
      ++i;
      Json arr = Json::array();
      ws();
      if (s[i] == ']') { ++i; return arr; }
      while (true) {
        arr.push_back(value());
        ws();
        if (s[i] == ',') { ++i; continue; }
        if (s[i] == ']') { ++i; return arr; }
        throw std::runtime_error("expected ',' or ']' in JSON");
      }
    }
    if (c == '"') return Json(string());
    if (starts("true")) { i += 4; return Json(true); }
    if (starts("false")) { i += 5; return Json(false); }
    if (starts("null")) { i += 4; return Json(); }
    if (starts("NaN")) { i += 3; return Json(std::nan("")); }
    if (starts("Infinity")) { i += 8; return Json(HUGE_VAL); }
    if (starts("-Infinity")) { i += 9; return Json(-HUGE_VAL); }
    size_t j = i;
    bool is_float = false;
    while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '-' || s[j] == '+' ||
                            s[j] == '.' || s[j] == 'e' || s[j] == 'E')) {
      if (s[j] == '.' || s[j] == 'e' || s[j] == 'E') is_float = true;
      ++j;
    }
    const std::string token = s.substr(i, j - i);
    i = j;
    if (is_float) return Json(std::stod(token));
    return Json(static_cast<long long>(std::stoll(token)));
  }
  std::string string() {
    if (s[i] != '"') throw std::runtime_error("expected string in JSON");
    ++i;
    std::string out;
    while (s[i] != '"') {
      if (s[i] == '\\') {
        ++i;
        const char e = s[i++];
        if (e == 'n') out += '\n';
        else if (e == 't') out += '\t';
        else if (e == 'r') out += '\r';
        else if (e == 'u') {
          const int code = std::stoi(s.substr(i, 4), nullptr, 16);
          i += 4;
          if (code < 0x80) out += static_cast<char>(code);
          else if (code < 0x800) {
            out += static_cast<char>(0xC0 | (code >> 6));
            out += static_cast<char>(0x80 | (code & 0x3F));
          } else {
            out += static_cast<char>(0xE0 | (code >> 12));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
          }
        } else out += e;
      } else {
        out += s[i++];
      }
    }
    ++i;
    return out;
  }
};
}  // namespace

Json Json::parse(const std::string& text) {
  Parser p{text};
  return p.value();
}

void write_json(const std::filesystem::path& path, const Json& value, bool sort_keys) {
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path.string());
  out << value.dump(2, sort_keys) << "\n";
}

Json read_json(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  std::stringstream buffer;
  buffer << in.rdbuf();
  return Json::parse(buffer.str());
}

std::string csv_text(const CsvCell& cell) {
  if (std::holds_alternative<std::string>(cell)) return std::get<std::string>(cell);
  if (std::holds_alternative<double>(cell)) return py_repr(std::get<double>(cell));
  if (std::holds_alternative<std::int64_t>(cell)) return std::to_string(std::get<std::int64_t>(cell));
  return std::get<bool>(cell) ? "True" : "False";
}

namespace {
// Quote as the csv module does (QUOTE_MINIMAL).
std::string csv_field(const std::string& s) {
  if (s.find_first_of(",\"\r\n") == std::string::npos) return s;
  std::string out = "\"";
  for (char c : s) out += (c == '"') ? std::string("\"\"") : std::string(1, c);
  return out + "\"";
}
}  // namespace

void write_csv(const std::filesystem::path& path, const std::vector<std::string>& fields,
               const std::vector<CsvRow>& rows) {
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path.string());
  for (size_t i = 0; i < fields.size(); ++i) out << (i ? "," : "") << csv_field(fields[i]);
  out << "\r\n";
  for (const auto& row : rows) {
    for (const auto& key : row)
      if (std::find(fields.begin(), fields.end(), key.first) == fields.end())
        throw std::invalid_argument("dict contains fields not in fieldnames: " + key.first);
    for (size_t i = 0; i < fields.size(); ++i) {
      const auto it = row.find(fields[i]);
      out << (i ? "," : "") << (it == row.end() ? "" : csv_field(csv_text(it->second)));
    }
    out << "\r\n";
  }
}

CsvTable read_csv(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  std::vector<std::vector<std::string>> records;
  std::vector<std::string> record;
  std::string field;
  bool quoted = false, any = false;
  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (quoted) {
      if (c == '"' && i + 1 < text.size() && text[i + 1] == '"') { field += '"'; ++i; }
      else if (c == '"') quoted = false;
      else field += c;
    } else if (c == '"') {
      quoted = true;
      any = true;
    } else if (c == ',') {
      record.push_back(field);
      field.clear();
      any = true;
    } else if (c == '\r' || c == '\n') {
      if (c == '\r' && i + 1 < text.size() && text[i + 1] == '\n') ++i;
      if (any || !field.empty()) {
        record.push_back(field);
        records.push_back(record);
      }
      record.clear();
      field.clear();
      any = false;
    } else {
      field += c;
      any = true;
    }
  }
  if (any || !field.empty()) {
    record.push_back(field);
    records.push_back(record);
  }
  CsvTable table;
  if (records.empty()) return table;
  table.fields = records[0];
  for (size_t r = 1; r < records.size(); ++r) {
    std::map<std::string, std::string> row;
    for (size_t c = 0; c < table.fields.size() && c < records[r].size(); ++c) row[table.fields[c]] = records[r][c];
    table.rows.push_back(row);
  }
  return table;
}

std::string utc_timestamp() {
  using namespace std::chrono;
  const auto now = system_clock::now();
  const auto micros = duration_cast<microseconds>(now.time_since_epoch()).count() % 1000000;
  const std::time_t t = system_clock::to_time_t(now);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
  char out[96];
  std::snprintf(out, sizeof(out), "%s.%06lld+00:00", buf, static_cast<long long>(micros));
  return out;
}

}  // namespace regbie
