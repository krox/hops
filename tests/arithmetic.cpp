#include "catch2/catch_test_macros.hpp"
#include "fmt/format.h"
#include "fmt/std.h"
#include "hops/hops.h"
#include <random>

TEST_CASE("basic arithmetic", "[hops]")
{
	// init cuda, select device, create context
	hops::init();
	atexit(hops::finalize);

	size_t n = 1 << 18;
	auto hX = std::vector<float>(n);
	auto hY = std::vector<float>(n);

	// fill with random data
	auto rng = std::mt19937(42);
	auto dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
	for (size_t i = 0; i < n; ++i)
	{
		hX[i] = dist(rng);
		hY[i] = dist(rng);
	}

	auto dX = hops::device_buffer<float>::from_host(hX);
	auto dY = hops::device_buffer<float>::from_host(hY);
	auto dOut = hops::device_buffer<float>(n);

	hops::mul(dOut.view(), 2.5f, dX.view(), dY.view());
	hops::add_mul(dOut.view().step(2), -1.0, dX.view().step(2),
	              dY.view().step(2));

	auto hOut = dOut.to_host();

	for (size_t i = 0; i < n; ++i)
	{
		auto expected = (i % 2 == 0 ? 1.5f : 2.5f) * hX[i] * hY[i];
		REQUIRE(std::abs(hOut[i] - expected) < 1e-6);
	}
}

TEST_CASE("complex arithmetic", "[hops]")
{
	hops::init();
	atexit(hops::finalize);

	size_t n = 1 << 18;
	auto hX = std::vector<std::complex<float>>(n);
	auto hY = std::vector<std::complex<float>>(n);

	auto rng = std::mt19937(42);
	auto dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
	for (size_t i = 0; i < n; ++i)
	{
		hX[i] = {dist(rng), dist(rng)};
		hY[i] = {dist(rng), dist(rng)};
	}

	auto dX = hops::device_buffer<std::complex<float>>::from_host(hX);
	auto dY = hops::device_buffer<std::complex<float>>::from_host(hY);
	auto dOut = hops::device_buffer<std::complex<float>>(n);

	hops::mul<std::complex<float>>(dOut.view(), 2.5, dX.view(), dY.view());
	hops::add_mul<std::complex<float>>(dOut.view().step(2), -1.0f,
	                                   dX.view().step(2), dY.view().step(2));

	auto hOut = dOut.to_host();

	for (size_t i = 0; i < n; ++i)
	{
		auto expected = (i % 2 == 0 ? 1.5f : 2.5f) * hX[i] * hY[i];
		REQUIRE(std::abs(hOut[i].real() - expected.real()) < 1e-6);
		REQUIRE(std::abs(hOut[i].imag() - expected.imag()) < 1e-6);
	}
}
