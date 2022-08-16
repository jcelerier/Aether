#include "aether_dsp.hpp"

#include "constants.hpp"
#include "parameters.hpp"
#include "utils.hpp"
#include "math.hpp"

#include <cmath>

#include <algorithm>
#include <utility>

namespace Aether
{

namespace
{
inline float dBtoGain(float db) noexcept
{
  return std::pow(10.f, db / 20.f);
}
}

DSP::DSP(float rate)
    : m_l_predelay(rate)
    , m_r_predelay(rate)
    , m_l_early_filters(rate)
    , m_r_early_filters(rate)
    , m_l_early_multitap(rate)
    , m_r_early_multitap(rate)
    , m_l_early_diffuser(rate, rng)
    , m_r_early_diffuser(rate, rng)
    , m_l_late_rev(rate, rng)
    , m_r_late_rev(rate, rng)
    , m_rate{rate}
{
  for(size_t i = 0; i != param_targets.size(); ++i)
    param_targets[i] = params[i] = parameter_infos[i + 6].dflt;

  for(bool& modified : params_modified)
    modified = true;

  param_smooth.mix = 50.f;

  param_smooth.dry_level = 50.f;
  param_smooth.predelay_level = 50.f;
  param_smooth.early_level = 50.f;
  param_smooth.late_level = 50.f;

  param_smooth.width = 50.f;
  param_smooth.predelay = 5000.f;

  param_smooth.early_tap_mix = 50.f;
  param_smooth.early_tap_decay = 25.f;
  param_smooth.early_tap_length = 4000.f;

  param_smooth.early_diffusion_delay = 5000.f;
  param_smooth.early_diffusion_mod_depth = 1000.f;
  param_smooth.early_diffusion_feedback = 500.f;

  param_smooth.late_delay = 5000.f;
  param_smooth.late_delay_mod_depth = 1000.f;
  param_smooth.late_delay_line_feedback = 50.f;

  param_smooth.late_diffusion_delay = 5000.f;
  param_smooth.late_diffusion_mod_depth = 2000.f;
  param_smooth.late_diffusion_feedback = 500.f;

  param_smooth.seed_crossmix = 5000.f;

  for(float& smooth : param_smooth)
  {
    constexpr float pi = constants::pi_v<float>;
    if(smooth != 0.f)
      smooth = std::exp(-2 * pi / (0.0001f * smooth * rate));
  }
}

void Object::prepare(halp::setup s)
{
  if(dsp.m_rate != s.rate)
  {
    std::destroy_at(&dsp);
    std::construct_at(&dsp, s.rate);
  }

  auto ptr = dsp.param_ports.data();

  *ptr++ = &inputs.mix.value;
  *ptr++ = &inputs.dry_level.value;
  *ptr++ = &inputs.predelay_level.value;
  *ptr++ = &inputs.early_level.value;
  *ptr++ = &inputs.late_level.value;
  *ptr++ = &inputs.interpolate.value;
  *ptr++ = &inputs.width.value;
  *ptr++ = &inputs.predelay.value;
  *ptr++ = &inputs.early_low_cut_enabled.value;
  *ptr++ = &inputs.early_low_cut_cutoff.value;
  *ptr++ = &inputs.early_high_cut_enabled.value;
  *ptr++ = &inputs.early_high_cut_cutoff.value;
  *ptr++ = &inputs.early_taps.value;
  *ptr++ = &inputs.early_tap_length.value;
  *ptr++ = &inputs.early_tap_mix.value;
  *ptr++ = &inputs.early_tap_decay.value;
  *ptr++ = &inputs.early_diffusion_stages.value;
  *ptr++ = &inputs.early_diffusion_delay.value;
  *ptr++ = &inputs.early_diffusion_mod_depth.value;
  *ptr++ = &inputs.early_diffusion_mod_rate.value;
  *ptr++ = &inputs.early_diffusion_feedback.value;
  *ptr++ = &inputs.late_order.value;
  *ptr++ = &inputs.late_delay_lines.value;
  *ptr++ = &inputs.late_delay.value;
  *ptr++ = &inputs.late_delay_mod_depth.value;
  *ptr++ = &inputs.late_delay_mod_rate.value;
  *ptr++ = &inputs.late_delay_line_feedback.value;
  *ptr++ = &inputs.late_diffusion_stages.value;
  *ptr++ = &inputs.late_diffusion_delay.value;
  *ptr++ = &inputs.late_diffusion_mod_depth.value;
  *ptr++ = &inputs.late_diffusion_mod_rate.value;
  *ptr++ = &inputs.late_diffusion_feedback.value;
  *ptr++ = &inputs.late_low_shelf_enabled.value;
  *ptr++ = &inputs.late_low_shelf_cutoff.value;
  *ptr++ = &inputs.late_low_shelf_gain.value;
  *ptr++ = &inputs.late_high_shelf_enabled.value;
  *ptr++ = &inputs.late_high_shelf_cutoff.value;
  *ptr++ = &inputs.late_high_shelf_gain.value;
  *ptr++ = &inputs.late_high_cut_enabled.value;
  *ptr++ = &inputs.late_high_cut_cutoff.value;
  *ptr++ = &inputs.seed_crossmix.value;
  *ptr++ = &inputs.tap_seed.value;
  *ptr++ = &inputs.early_diffusion_seed.value;
  *ptr++ = &inputs.delay_seed.value;
  *ptr++ = &inputs.late_diffusion_seed.value;
  *ptr++ = &inputs.early_diffusion_drive.value;
  *ptr++ = &inputs.late_diffusion_drive.value;

  dsp.apply_parameters();
}

void Object::operator()(uint32_t n_samples) noexcept
{
    dsp.update_parameter_targets();
    auto audio_in_left = inputs.audio[0];
    auto audio_in_right = inputs.audio[1];
    auto audio_out_left = outputs.audio[0];
    auto audio_out_right = outputs.audio[1];
    for(uint32_t sample = 0; sample < n_samples; ++sample)
    {
        dsp.update_parameters();

        // Dry
        float dry_level = dsp.params.dry_level / 100.f;
        float dry_left = audio_in_left[sample];
        float dry_right = audio_in_right[sample];
        {
            audio_out_left[sample] = dry_level * dry_left;
            audio_out_right[sample] = dry_level * dry_right;
        }

        // Predelay
        float predelay_level = dsp.params.predelay_level / 100.f;
        float predelay_left = dry_left;
        float predelay_right = dry_right;
        {
            float width = 0.5f - dsp.params.width / 200.f;
            predelay_left = dry_left + width * (dry_right - dry_left);
            predelay_right = dry_right - width * (dry_right - dry_left);

            // predelay in samples
            uint32_t delay = static_cast<uint32_t>(dsp.params.predelay / 1000.f * dsp.m_rate);
            predelay_left = dsp.m_l_predelay.push(predelay_left, delay);
            predelay_right = dsp.m_r_predelay.push(predelay_right, delay);

            audio_out_left[sample] += predelay_level * predelay_left;
            audio_out_right[sample] += predelay_level * predelay_right;
        }

        // Early Reflections
        float early_level = dsp.params.early_level / 100.f;
        float early_left = predelay_left;
        float early_right = predelay_right;
        {
            // Filtering
            if(dsp.params.early_low_cut_enabled > 0.f)
            {
                early_left = dsp.m_l_early_filters.highpass.push(early_left);
                early_right = dsp.m_r_early_filters.highpass.push(early_right);
            }

            if(dsp.params.early_high_cut_enabled > 0.f)
            {
                early_left = dsp.m_l_early_filters.lowpass.push(early_left);
                early_right = dsp.m_r_early_filters.lowpass.push(early_right);
            }

            { // multitap delay
                uint32_t taps = static_cast<uint32_t>(dsp.params.early_taps);
                float length = dsp.params.early_tap_length / 1000.f * dsp.m_rate;

                float multitap_left = dsp.m_l_early_multitap.push(early_left, taps, length);
                float multitap_right = dsp.m_r_early_multitap.push(early_right, taps, length);

                float tap_mix = dsp.params.early_tap_mix / 100.f;
                early_left += tap_mix * (multitap_left - early_left);
                early_right += tap_mix * (multitap_right - early_right);
      }

      { // allpass diffuser
        AllpassDiffuser<float>::PushInfo info = {};
        info.stages = static_cast<uint32_t>(dsp.params.early_diffusion_stages);
        info.feedback = dsp.params.early_diffusion_feedback;
        info.interpolate = true;

        early_left = dsp.m_l_early_diffuser.push(early_left, info);
        early_right = dsp.m_r_early_diffuser.push(early_right, info);
      }

      audio_out_left[sample] += early_level * early_left;
      audio_out_right[sample] += early_level * early_right;
    }

    // Late Reverberations
    float late_level = dsp.params.late_level / 100.f;
    float late_left = early_left;
    float late_right = early_right;
    {
      AllpassDiffuser<double>::PushInfo diffuser_info = {};
      diffuser_info.stages = static_cast<uint32_t>(dsp.params.late_diffusion_stages);
      diffuser_info.feedback = dsp.params.late_diffusion_feedback;
      diffuser_info.interpolate = dsp.params.interpolate > 0;

      Delayline::Filters::PushInfo damping_info = {};
      damping_info.ls_enable = dsp.params.late_low_shelf_enabled > 0;
      damping_info.hs_enable = dsp.params.late_high_shelf_enabled > 0;
      damping_info.hc_enable = dsp.params.late_high_cut_enabled > 0;

      Delayline::PushInfo push_info = {};
      push_info.order = static_cast<Delayline::Order>(dsp.params.late_order);
      push_info.diffuser_info = diffuser_info;
      push_info.damping_info = damping_info;

      late_left = dsp.m_l_late_rev.push(late_left, push_info);
      late_right = dsp.m_r_late_rev.push(late_right, push_info);
      audio_out_left[sample] += late_level * late_left;
      audio_out_right[sample] += late_level * late_right;
    }

    {
      float mix = dsp.params.mix / 100.f;
      audio_out_left[sample] = math::lerp(dry_left, audio_out_left[sample], mix);
      audio_out_right[sample] = math::lerp(dry_right, audio_out_right[sample], mix);
    }
  }
}

void DSP::update_parameter_targets() noexcept
{
  for(size_t p = 0; p < param_targets.size(); ++p)
  {
    param_targets[p] = std::clamp(
        param_ports[p] ? *param_ports[p] : parameter_infos[p + 6].dflt,
        parameter_infos[p + 6].min, parameter_infos[p + 6].max);
  }
}

void DSP::update_parameters() noexcept
{
  for(size_t p = 0; p < param_ports.size(); ++p)
  {
    const float new_value
        = param_targets[p] - param_smooth[p] * (param_targets[p] - params[p]);
    params_modified[p] = (new_value != params[p]);
    params[p] = new_value;
  }

  apply_parameters();
}

void DSP::apply_parameters() noexcept
{
  // Early Reflections

  // Filters
  if(params_modified.early_low_cut_cutoff)
  {
    float cutoff = params.early_low_cut_cutoff;
    m_l_early_filters.highpass.set_cutoff(cutoff);
    m_r_early_filters.highpass.set_cutoff(cutoff);
  }
  if(params_modified.early_high_cut_cutoff)
  {
    float cutoff = params.early_high_cut_cutoff;
    m_l_early_filters.lowpass.set_cutoff(cutoff);
    m_r_early_filters.lowpass.set_cutoff(cutoff);
  }

  // Multitap Delay
  if(params_modified.early_tap_decay)
  {
    float decay = params.early_tap_decay;
    m_l_early_multitap.set_decay(decay);
    m_r_early_multitap.set_decay(decay);
  }
  if(params_modified.seed_crossmix)
  {
    float crossmix = params.seed_crossmix / 200.f;
    m_l_early_multitap.set_seed_crossmix(1.f - crossmix);
    m_r_early_multitap.set_seed_crossmix(0.f + crossmix);
  }
  if(params_modified.tap_seed)
  {
    uint32_t seed = static_cast<uint32_t>(params.tap_seed);
    m_l_early_multitap.set_seed(seed);
    m_r_early_multitap.set_seed(seed);
  }

  // Diffuser
  if(params_modified.early_diffusion_drive)
  {
    float drive = params.early_diffusion_drive == -12
                      ? 0
                      : dBtoGain(params.early_diffusion_drive);
    m_l_early_diffuser.set_drive(drive);
    m_r_early_diffuser.set_drive(drive);
  }
  if(params_modified.early_diffusion_delay)
  {
    float delay = m_rate * params.early_diffusion_delay / 1000.f;
    m_l_early_diffuser.set_delay(delay);
    m_r_early_diffuser.set_delay(delay);
  }
  if(params_modified.early_diffusion_mod_depth)
  {
    float mod_depth = m_rate * params.early_diffusion_mod_depth / 1000.f;
    m_l_early_diffuser.set_mod_depth(mod_depth);
    m_r_early_diffuser.set_mod_depth(mod_depth);
  }
  if(params_modified.early_diffusion_mod_rate)
  {
    float rate = params.early_diffusion_mod_rate / m_rate;
    m_l_early_diffuser.set_mod_rate(rate);
    m_r_early_diffuser.set_mod_rate(rate);
  }
  if(params_modified.seed_crossmix)
  {
    float crossmix = params.seed_crossmix / 200.f;
    m_l_early_diffuser.set_seed_crossmix(1.f - crossmix);
    m_r_early_diffuser.set_seed_crossmix(0.f + crossmix);
  }
  if(params_modified.early_diffusion_seed)
  {
    uint32_t seed = static_cast<uint32_t>(params.early_diffusion_seed);
    m_l_early_diffuser.set_seed(seed);
    m_r_early_diffuser.set_seed(seed);
  }

  // Late Reverberations

  // General
  if(params_modified.seed_crossmix)
  {
    float crossmix = params.seed_crossmix / 200.f;
    m_l_late_rev.set_seed_crossmix(1.f - crossmix);
    m_r_late_rev.set_seed_crossmix(0.f + crossmix);
  }
  if(params_modified.late_delay_lines)
  {
    uint32_t lines = static_cast<uint32_t>(params.late_delay_lines);
    m_l_late_rev.set_delay_lines(lines);
    m_r_late_rev.set_delay_lines(lines);
  }

  // Modulated Delay
  if(params_modified.late_delay)
  {
    float delay = m_rate * params.late_delay / 1000.f;
    m_l_late_rev.set_delay(delay);
    m_r_late_rev.set_delay(delay);
  }
  if(params_modified.late_delay_mod_depth)
  {
    float mod_depth = m_rate * params.late_delay_mod_depth / 1000.f;
    m_l_late_rev.set_delay_mod_depth(mod_depth);
    m_r_late_rev.set_delay_mod_depth(mod_depth);
  }
  if(params_modified.late_delay_mod_rate)
  {
    float mod_rate = params.late_delay_mod_rate / m_rate;
    m_l_late_rev.set_delay_mod_rate(mod_rate);
    m_r_late_rev.set_delay_mod_rate(mod_rate);
  }
  if(params_modified.late_delay_line_feedback)
  {
    float feedback = params.late_delay_line_feedback;
    m_l_late_rev.set_delay_feedback(feedback);
    m_r_late_rev.set_delay_feedback(feedback);
  }
  if(params_modified.delay_seed)
  {
    uint32_t seed = static_cast<uint32_t>(params.delay_seed);
    m_l_late_rev.set_delay_seed(seed);
    m_r_late_rev.set_delay_seed(seed);
  }

  // Diffuser
  if(params_modified.late_diffusion_drive)
  {
    float drive
        = params.late_diffusion_drive == -12 ? 0 : dBtoGain(params.late_diffusion_drive);
    m_l_late_rev.set_diffusion_drive(drive);
    m_r_late_rev.set_diffusion_drive(drive);
  }
  if(params_modified.late_diffusion_delay)
  {
    float delay = m_rate * params.late_diffusion_delay / 1000.f;
    m_l_late_rev.set_diffusion_delay(delay);
    m_r_late_rev.set_diffusion_delay(delay);
  }
  if(params_modified.late_diffusion_mod_depth)
  {
    float depth = m_rate * params.late_diffusion_mod_depth / 1000.f;
    m_l_late_rev.set_diffusion_mod_depth(depth);
    m_r_late_rev.set_diffusion_mod_depth(depth);
  }
  if(params_modified.late_diffusion_mod_rate)
  {
    float rate = params.late_diffusion_mod_rate / m_rate;
    m_l_late_rev.set_diffusion_mod_rate(rate);
    m_r_late_rev.set_diffusion_mod_rate(rate);
  }
  if(params_modified.late_diffusion_seed)
  {
    uint32_t seed = static_cast<uint32_t>(params.late_diffusion_seed);
    m_l_late_rev.set_diffusion_seed(seed);
    m_r_late_rev.set_diffusion_seed(seed);
  }

  // Filters
  if(params_modified.late_low_shelf_cutoff)
  {
    float cutoff = params.late_low_shelf_cutoff;
    m_l_late_rev.set_low_shelf_cutoff(cutoff);
    m_r_late_rev.set_low_shelf_cutoff(cutoff);
  }
  if(params_modified.late_low_shelf_gain)
  {
    float gain = dBtoGain(params.late_low_shelf_gain);
    m_l_late_rev.set_low_shelf_gain(gain);
    m_r_late_rev.set_low_shelf_gain(gain);
  }
  if(params_modified.late_high_shelf_cutoff)
  {
    float cutoff = params.late_high_shelf_cutoff;
    m_l_late_rev.set_high_shelf_cutoff(cutoff);
    m_r_late_rev.set_high_shelf_cutoff(cutoff);
  }
  if(params_modified.late_high_shelf_gain)
  {
    float gain = dBtoGain(params.late_high_shelf_gain);
    m_l_late_rev.set_high_shelf_gain(gain);
    m_r_late_rev.set_high_shelf_gain(gain);
  }
  if(params_modified.late_high_cut_cutoff)
  {
    float cutoff = params.late_high_cut_cutoff;
    m_l_late_rev.set_high_cut_cutoff(cutoff);
    m_r_late_rev.set_high_cut_cutoff(cutoff);
  }
}
}
