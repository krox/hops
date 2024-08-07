#pragma once

// this is the main header file for the hops library. Typically should be the
// only one included by users of the library.

#include "hops/arithmetic.h"
#include "hops/error.h"
#include "hops/kernel.h"
#include "hops/memory.h"
#include <iostream>
#include <string>
#include <vector>

namespace hops {

// Initialize CUDA, select a device, and create context.
//   * verbose=1 also prints some info about the used device to stdout
//   * rationale: CUDA manuals advises strongly against mixing multiple contexts
//     in a single process, and nearly all cuda routines pick up the current
//     context automatically. So there is probably no point in exposing it
//     explicitly here.
//   * will throw if the specified device does not exist or cannot be selected
//   * only call once, not thread safe
void init(int device_id = 0, int verbose = 1);

// destroy context. (note: cuda itself stays initialized)
void finalize() noexcept;

// synchronize with the device
void sync();

} // namespace hops