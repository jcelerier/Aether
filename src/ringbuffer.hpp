#ifndef RINGBUFFER_HPP
#define RINGBUFFER_HPP

#include <algorithm>
#include <cstddef>

namespace Aether
{
template <class T>
struct Ringbuffer
{

  Ringbuffer()
      : Ringbuffer(0)
  {
  }
  explicit Ringbuffer(size_t sz)
      : size{sz}
      , buf{new T[sz]}
  {
    clear();
  }

  Ringbuffer(const Ringbuffer&) = delete;
  Ringbuffer& operator=(const Ringbuffer&) = delete;

  Ringbuffer(Ringbuffer&& other) noexcept:
      buf{}
    , size{}
    , end{}
  {
    swap(other);
  }

  Ringbuffer& operator=(Ringbuffer&& other) noexcept
  {
    swap(other);
    return *this;
  }

  ~Ringbuffer() { delete[] buf; }

  void push(T value) noexcept
  {
    ++end;
    end -= (end >= size ? size : 0);
    buf[end] = value;
  }

  void clear() noexcept { std::fill_n(buf, size, T()); }

  void swap(Ringbuffer& other) noexcept
  {
    std::swap(end, other.end);
    std::swap(size, other.size);
    std::swap(buf, other.buf);
  }

  size_t end = 0;
  size_t size;
  T* buf;
};
}
namespace std
{
template <class T>
inline void swap(Aether::Ringbuffer<T>& lhs, Aether::Ringbuffer<T>& rhs) noexcept
{
  lhs.swap(rhs);
}
}


#endif
