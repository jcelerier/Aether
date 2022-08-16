#pragma once

#include "delay.hpp"
#include "delayline.hpp"
#include "diffuser.hpp"
#include "filters.hpp"
#include "random.hpp"

#include <halp/audio.hpp>
#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string_view>

namespace Aether
{
class Object;
class DSP
{
  friend class Object;
  template <class T>
  struct Parameters
  {
    T mix;

    // mixer
    T dry_level;
    T predelay_level;
    T early_level;
    T late_level;

    // Global
    T interpolate;

    // predelay
    T width;
    T predelay;

    // early
    // filtering
    T early_low_cut_enabled;
    T early_low_cut_cutoff;
    T early_high_cut_enabled;
    T early_high_cut_cutoff;
    // multitap delay
    T early_taps;
    T early_tap_length;
    T early_tap_mix;
    T early_tap_decay;
    // diffusion
    T early_diffusion_stages;
    T early_diffusion_delay;
    T early_diffusion_mod_depth;
    T early_diffusion_mod_rate;
    T early_diffusion_feedback;

    // late
    T late_order;
    T late_delay_lines;
    // delay line
    T late_delay;
    T late_delay_mod_depth;
    T late_delay_mod_rate;
    T late_delay_line_feedback;
    // diffusion
    T late_diffusion_stages;
    T late_diffusion_delay;
    T late_diffusion_mod_depth;
    T late_diffusion_mod_rate;
    T late_diffusion_feedback;
    // Filter
    T late_low_shelf_enabled;
    T late_low_shelf_cutoff;
    T late_low_shelf_gain;
    T late_high_shelf_enabled;
    T late_high_shelf_cutoff;
    T late_high_shelf_gain;
    T late_high_cut_enabled;
    T late_high_cut_cutoff;

    // Seed
    T seed_crossmix;
    T tap_seed;
    T early_diffusion_seed;
    T delay_seed;
    T late_diffusion_seed;

    // Distortion
    T early_diffusion_drive;
    T late_diffusion_drive;

    T& operator[](size_t idx) noexcept { return data()[idx]; }
    const T& operator[](size_t idx) const noexcept { return data()[idx]; }

    T* data() noexcept { return reinterpret_cast<T*>(this); }
    const T* data() const noexcept { return reinterpret_cast<const T*>(this); }

    T* begin() noexcept { return data(); }
    const T* begin() const noexcept { return data(); }

    T* end() noexcept { return data() + size(); }
    const T* end() const noexcept { return data() + size(); }

    static constexpr size_t size() noexcept { return sizeof(Parameters<T>) / sizeof(T); }
  };
  Parameters<float> params = {};
  Parameters<float> param_targets = {};
  Parameters<float> param_smooth = {};
  Parameters<bool> params_modified = {};
  std::array<const float*, 47> param_ports = {};

  /*
      Member Functions
    */
public:
  explicit DSP(float rate);

  void operator()(uint32_t n_samples) noexcept;

private:
  Random::Xorshift64s rng{std::random_device{}()};

  // Predelay
  Delay m_l_predelay;
  Delay m_r_predelay;

  // Early
  struct Filters
  {
    explicit Filters(float rate)
        : lowpass(rate)
        , highpass(rate)
    {
    }
    Filters(Filters&& other) noexcept = default;
    Filters& operator=(Filters&& other) noexcept = default;

    Lowpass6dB<float> lowpass;
    Highpass6dB<float> highpass;
  };

  Filters m_l_early_filters;
  Filters m_r_early_filters;

  MultitapDelay m_l_early_multitap;
  MultitapDelay m_r_early_multitap;

  AllpassDiffuser<float> m_l_early_diffuser;
  AllpassDiffuser<float> m_r_early_diffuser;

  // Late
  LateRev m_l_late_rev;
  LateRev m_r_late_rev;

  float m_rate;

  // send audio data if ui is open
  bool ui_open = false;

  // Updates param_targets
  void update_parameter_targets() noexcept;
  // Updates params & params_modified then calls apply_parameters
  void update_parameters() noexcept;
  // Applies changes in params & params_modified to internal state
  void apply_parameters() noexcept;
};

class Object
{
public:
  halp_meta(name, "Aether")
  halp_meta(c_name, "aether_reverb")
  halp_meta(category, "Audio/Effects")
  halp_meta(description, "An algorithmic reverb based on Cloudseed.")
  halp_meta(uuid, "3ed27d12-ca4e-4d23-a1b6-198cef6ba198")
  halp_meta(author, "Dougal-s, ValdemarOrn")
  halp_meta(uri, "http://github.com/Dougal-s/Aether")

  DSP dsp{44100.};

  using range = halp::range;
  using irange = halp::irange;
  struct
  {
    halp::fixed_audio_bus<"Audio", float, 2> audio;

    halp::knob_f32<"Mix", range{0., 100., 100.}> mix;

    // mixer
    halp::knob_f32<"Dry", range{0., 100., 80.}> dry_level;
    halp::knob_f32<"Predelay level", range{0., 100., 20.}> predelay_level;
    halp::knob_f32<"Early level", range{0., 100., 10.}> early_level;
    halp::knob_f32<"Late level", range{0., 100., 20.}> late_level;

    // Global
    halp::toggle_f32<"Interpolate"> interpolate;

    // predelay
    halp::knob_f32<"Width", range{0., 100., 100.}> width;
    halp::knob_f32<"Predelay", range{0., 400., 20.}> predelay;

    // early
    // filtering
    halp::toggle_f32<"Early low cut enabled"> early_low_cut_enabled;
    halp::knob_f32<"Early Low Cut Cutoff", range{15, 22000, 15}> early_low_cut_cutoff;
    halp::toggle_f32<"Early high cut enabled"> early_high_cut_enabled;
    halp::knob_f32<"Early High Cut Cutoff", range{15, 22000, 20000}> early_high_cut_cutoff;

    // multitap delay
    halp::iknob_f32<"Early taps", irange{1, 50, 12}> early_taps;
    halp::knob_f32<"Early tap length", range{0, 500, 200}> early_tap_length;
    halp::knob_f32<"Early tap mix", range{0, 100,100}> early_tap_mix;
    halp::knob_f32<"Early Tap Decay", range{0, 1, 0.5f}> early_tap_decay;
    // diffusion
    halp::iknob_f32<"Early diffusion stages", irange{0, 8, 7}> early_diffusion_stages;
    halp::knob_f32<"Early diffusion delay", range{10, 100, 20}>early_diffusion_delay;
    halp::knob_f32<"Early diffusion mod depth", range{0, 3, 0}>early_diffusion_mod_depth;
    halp::knob_f32<"Early diffusion mod rate", range{0, 5, 1}>early_diffusion_mod_rate;
    halp::knob_f32<"Early diffusion feedback", range{0, 1, 0.7f}> early_diffusion_feedback;

    // late
    halp::iknob_f32<"Late order", irange{0, 1, 0}> late_order;
    halp::iknob_f32<"Late delay lines", irange{1, 12, 3}> late_delay_lines;
    // delay line
    halp::knob_f32<"Late delay", range{0.05f, 1000, 100}> late_delay;
    halp::knob_f32<"Late delay mod depth", range{0, 50, 0.2f}> late_delay_mod_depth;
    halp::knob_f32<"Late delay mod rate", range{0, 5, 0.2f}> late_delay_mod_rate;
    halp::knob_f32<"Late delay line feedback", range{0, 1, 0.7f}> late_delay_line_feedback;
    // diffusion
    halp::iknob_f32<"Late diffusion stages", irange{0, 8, 7}> late_diffusion_stages;
    halp::knob_f32<"Late diffusion delay", range{10, 100, 50}> late_diffusion_delay;
    halp::knob_f32<"Late diffusion mod depth", range{0, 3, 0.2f}> late_diffusion_mod_depth;
    halp::knob_f32<"Late diffusion mod rate", range{0, 5, 0.5f}> late_diffusion_mod_rate;
    halp::knob_f32<"Late diffusion feedback", range{0, 1, 0.7f}> late_diffusion_feedback;
    // Filter
    halp::toggle_f32<"Late low shelf enabled"> late_low_shelf_enabled;
    halp::knob_f32<"Late low shelf cutoff", range{15, 22000, 100}> late_low_shelf_cutoff;
    halp::knob_f32<"Late low shelf gain", range{-24, 0, -2}> late_low_shelf_gain;
    halp::toggle_f32<"Late high shelf enabled"> late_high_shelf_enabled;
    halp::knob_f32<"Late high shelf cutoff", range{15, 22000, 1500}> late_high_shelf_cutoff;
    halp::knob_f32<"Late high shelf gain", range{-24, 0, -3}> late_high_shelf_gain;
    halp::toggle_f32<"Late high cut enabled"> late_high_cut_enabled;
    halp::knob_f32<"Late high cut cutoff", range{15, 22000, 20000}> late_high_cut_cutoff;

    // Seed
    halp::knob_f32<"Seed crossmix", range{0, 100, 80}> seed_crossmix;

    halp::iknob_f32<"Tap seed", irange{1, 99999, 1}> tap_seed;
    halp::iknob_f32<"Early diffusion seed", irange{1, 99999, 1}> early_diffusion_seed;
    halp::iknob_f32<"Delay seed", irange{1, 99999, 1}> delay_seed;
    halp::iknob_f32<"Late diffusion seed", irange{1, 99999, 1}> late_diffusion_seed;

    // Distortion
    halp::knob_f32<"Early diffusion drive", range{-12, 12, -12}> early_diffusion_drive;
    halp::knob_f32<"Late diffusion drive", range{-12, 12, -12}> late_diffusion_drive;
  } inputs;

  struct
  {
    halp::fixed_audio_bus<"Audio", float, 2> audio;
  } outputs;

  void prepare(halp::setup s);

  void operator()(uint32_t n_samples) noexcept;
};

}
