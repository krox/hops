#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace hops {

enum class ParameterKind
{
	raw,
	in,
	out,
	inout,
};

struct Parameter
{
	ParameterKind kind;
	std::string type;
	std::string name;
};

class Signature
{
	std::vector<Parameter> params_;

  public:
	Signature() = default;

	std::vector<Parameter> const &params() const { return params_; }

	Signature &raw(std::string_view type, std::string_view name)
	{
		params_.push_back({.kind = ParameterKind::raw,
		                   .type = std::string(type),
		                   .name = std::string(name)});
		return *this;
	}
	Signature &in(std::string_view type, std::string_view name)
	{
		params_.push_back({.kind = ParameterKind::in,
		                   .type = std::string(type),
		                   .name = std::string(name)});
		return *this;
	}
	Signature &out(std::string_view type, std::string_view name)
	{
		params_.push_back({.kind = ParameterKind::out,
		                   .type = std::string(type),
		                   .name = std::string(name)});
		return *this;
	}
	Signature &inout(std::string_view type, std::string_view name)
	{
		params_.push_back({.kind = ParameterKind::inout,
		                   .type = std::string(type),
		                   .name = std::string(name)});
		return *this;
	}
};

} // namespace hops