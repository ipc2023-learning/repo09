//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

#if !defined(FORMALISM_MURMURHASH3_HPP_)
#define FORMALISM_MURMURHASH3_HPP_

//-----------------------------------------------------------------------------
// Platform-specific functions and macros

// Microsoft Visual Studio

#if defined(_MSC_VER) && (_MSC_VER < 1600)

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned __int64 uint64_t;

#else  // defined(_MSC_VER)

#include <stdint.h>

#endif  // !defined(_MSC_VER)

//-----------------------------------------------------------------------------

void murmurhash3_128(const void* key, int len, uint32_t seed, uint64_t* out);

//-----------------------------------------------------------------------------

#endif  // FORMALISM_MURMURHASH3_HPP_