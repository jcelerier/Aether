#include <benchmark/benchmark.h>

#include "DSP/aether_dsp.hpp"

static void bm_aether(benchmark::State& state) {
	static constexpr size_t buffer_size = 1024;
	float* in_buf = new float[buffer_size];
	float* out_buf = new float[buffer_size];

	Aether::DSP dsp(48000);
	dsp.ports.audio_in_left = in_buf;
	dsp.ports.audio_in_right = in_buf;
	dsp.ports.audio_out_left = out_buf;
	dsp.ports.audio_out_right = out_buf;

	float mix = 0.f;
	float dry_level = 0.f;
	float predelay_level = 0.f;
	float early_level = 0.f;
	float late_level = 0.f;
	float interpolate = 0.f;
	float width = 100.f;
	float predelay = 0.f;
	float early_low_cut_enabled = 0.f;
	float early_low_cut_cutoff = 0.f;
	float early_high_cut_enabled = 0.f;
	float early_high_cut_cutoff = 0.f;
	float early_taps = 5.f;
	float early_tap_length = 0.f;
	float early_tap_mix = 0.f;
	float early_tap_decay = 0.f;
	float early_diffusion_stages = 0.f;
	float early_diffusion_delay = 50.f;
	float early_diffusion_mod_depth = 0.f;
	float early_diffusion_mod_rate = 0.f;
	float early_diffusion_feedback = 0.f;
	float late_order = 0.f;
	float late_delay_lines = 5.f;
	float late_delay = 5.f;
	float late_delay_mod_depth = 0.f;
	float late_delay_mod_rate = 0.f;
	float late_delay_line_feedback = 0.f;
	float late_diffusion_stages = 0.f;
	float late_diffusion_delay = 50.f;
	float late_diffusion_mod_depth = 0.f;
	float late_diffusion_mod_rate = 0.f;
	float late_diffusion_feedback = 0.f;
	float late_low_shelf_enabled = 0.f;
	float late_low_shelf_cutoff = 0.f;
	float late_low_shelf_gain = 0.f;
	float late_high_shelf_enabled = 0.f;
	float late_high_shelf_cutoff = 0.f;
	float late_high_shelf_gain = 0.f;
	float late_high_cut_enabled = 0.f;
	float late_high_cut_cutoff = 0.f;
	float seed_crossmix = 0.f;
	float tap_seed = 10.f;
	float early_diffusion_seed = 10.f;
	float delay_seed = 10.f;
	float late_diffusion_seed = 10.f;

	dsp.param_ports = {
		&mix,
		&dry_level,
		&predelay_level,
		&early_level,
		&late_level,
		&interpolate,
		&width,
		&predelay,
		&early_low_cut_enabled,
		&early_low_cut_cutoff,
		&early_high_cut_enabled,
		&early_high_cut_cutoff,
		&early_taps,
		&early_tap_length,
		&early_tap_mix,
		&early_tap_decay,
		&early_diffusion_stages,
		&early_diffusion_delay,
		&early_diffusion_mod_depth,
		&early_diffusion_mod_rate,
		&early_diffusion_feedback,
		&late_order,
		&late_delay_lines,
		&late_delay,
		&late_delay_mod_depth,
		&late_delay_mod_rate,
		&late_delay_line_feedback,
		&late_diffusion_stages,
		&late_diffusion_delay,
		&late_diffusion_mod_depth,
		&late_diffusion_mod_rate,
		&late_diffusion_feedback,
		&late_low_shelf_enabled,
		&late_low_shelf_cutoff,
		&late_low_shelf_gain,
		&late_high_shelf_enabled,
		&late_high_shelf_cutoff,
		&late_high_shelf_gain,
		&late_high_cut_enabled,
		&late_high_cut_cutoff,
		&seed_crossmix,
		&tap_seed,
		&early_diffusion_seed,
		&delay_seed,
		&late_diffusion_seed
	};

	// process parameter changes
	dsp.process(buffer_size);

	for (auto _ : state)
		dsp.process(buffer_size);

	delete[] in_buf;
	delete[] out_buf;
}

BENCHMARK(bm_aether)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
