#pragma once
#define NOMINMAX

#include <atomic>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include "FFTConvolver.h"

const size_t max_ir_size = 32768;
const size_t MAX_AUDIO_FRAMES = 4096; 

class AudioEngine {
public:
    enum RoomType { CATHEDRAL, CORRIDOR };

    struct State {
        std::atomic<float> lx{0.f}, ly{0.f}, lz{0.f};
        std::atomic<uint16_t> active_ir{0};
        std::atomic<float> blend_t{0.f};
        std::atomic<float> wet_level{0.f};
        std::atomic<int>   conv_length{2048};
    };

    State state;
    AudioEngine() : device_initialized(false), current_gain(0), current_blend(0), current_wet(0) {}
    ~AudioEngine() { shutdown(); }

    bool init(const char* source_path, const char* ir_cathedral, const char* ir_corridor) {
        ma_decoder_config dec_config = ma_decoder_config_init_default();
        dec_config.format = ma_format_f32; 

        if (ma_decoder_init_file(source_path, &dec_config, &decoder_source) != MA_SUCCESS) {
            std::cerr << "[audio engine] failed to load source: " << source_path << "\n";
            return false;
        }
        ma_data_source_set_looping(&decoder_source, MA_TRUE);
        
        load_ir_buffer(ir_cathedral, 1.5f, CATHEDRAL);
        load_ir_buffer(ir_corridor, 8.0f, CORRIDOR);

        ma_device_config config = ma_device_config_init(ma_device_type_playback);
        config.playback.format   = ma_format_f32; 
        config.playback.channels = decoder_source.outputChannels;
        config.sampleRate        = decoder_source.outputSampleRate;
        config.dataCallback      = data_callback_static;
        config.pUserData         = this;

        if (ma_device_init(NULL, &config, &device) != MA_SUCCESS) return false;
        device_initialized = true;
        ma_device_start(&device);
        return true;
    }

    void shutdown() {
        if (device_initialized) { ma_device_uninit(&device); ma_decoder_uninit(&decoder_source); device_initialized = false; }
    }

private:
    ma_device device;
    ma_decoder decoder_source;
    bool device_initialized;
    
    fftconvolver::FFTConvolver convolverCath;
    fftconvolver::FFTConvolver convolverCorr;
    
    const float sx = 12.5f, sy = 2.0f, sz = 25.0f;
    float current_gain, current_blend, current_wet;

    
    float rt_dry_mono[MAX_AUDIO_FRAMES] = {0};
    float rt_fft_cath[MAX_AUDIO_FRAMES] = {0};
    float rt_fft_corr[MAX_AUDIO_FRAMES] = {0};

    static void data_callback_static(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
        ((AudioEngine*)pDevice->pUserData)->process_audio((float*)pOutput, frameCount, pDevice->playback.channels);
    }

    void process_audio(float* pOutput, ma_uint32 frameCount, ma_uint32 channels) {
        ma_uint32 processFrames = std::min(frameCount, (ma_uint32)MAX_AUDIO_FRAMES);
        
        ma_uint64 framesRead = 0;
        ma_data_source_read_pcm_frames(&decoder_source, pOutput, processFrames, &framesRead);
        if (framesRead < processFrames) std::memset(pOutput + (framesRead * channels), 0, (processFrames - framesRead) * channels * sizeof(float));

        float target_lx = state.lx.load(std::memory_order_relaxed);
        float target_ly = state.ly.load(std::memory_order_relaxed);
        float target_lz = state.lz.load(std::memory_order_relaxed);
        float target_blend = state.blend_t.load(std::memory_order_relaxed);
        float target_wet = state.wet_level.load(std::memory_order_relaxed);
        
        float dx = target_lx - sx, dy = target_ly - sy, dz = target_lz - sz;
        float target_gain = std::clamp(1.0f / (1.0f + (std::sqrt(dx*dx + dy*dy + dz*dz) * 0.2f)), 0.1f, 1.0f);

        
        for (ma_uint32 i = 0; i < processFrames; ++i) {
            current_gain += (target_gain - current_gain) * 0.005f;
            current_blend += (target_blend - current_blend) * 0.005f;
            current_wet += (target_wet - current_wet) * 0.005f;

            rt_dry_mono[i] = pOutput[i * channels] * current_gain; 
        }

        
        convolverCath.process(rt_dry_mono, rt_fft_cath, processFrames);
        convolverCorr.process(rt_dry_mono, rt_fft_corr, processFrames);

        
        for (ma_uint32 i = 0; i < processFrames * channels; i += channels) {
            ma_uint32 frame_idx = i / channels;

            float dry_sample = rt_dry_mono[frame_idx];
            float wet_signal = (rt_fft_cath[frame_idx] * (1.0f - current_blend)) + (rt_fft_corr[frame_idx] * current_blend);
            float final_mix = (dry_sample * (1.0f - current_wet)) + (wet_signal * current_wet);

            for (ma_uint32 c = 0; c < channels; ++c) {
                pOutput[i + c] = final_mix;
            }
        }
    }

    void load_ir_buffer(const char* path, float synthetic_decay, RoomType type) {
        std::vector<float> ir_raw(max_ir_size, 0.0f);
        size_t read_len = 0; 

        ma_decoder temp_decoder;
        ma_decoder_config ir_config = ma_decoder_config_init_default();
        ir_config.format = ma_format_f32; 
        ir_config.channels = 1; 
        
        if (ma_decoder_init_file(path, &ir_config, &temp_decoder) != MA_SUCCESS) {
            std::cout << "[audio engine] generating synthetic ir for: " << path << "\n";
            for (size_t i = 0; i < max_ir_size; ++i) {
                float noise = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                ir_raw[i] = noise * std::exp(-synthetic_decay * (float)i / 48000.0f);
            }
            read_len = max_ir_size;
        } else {
            ma_uint64 total_frames = 0;
            if (ma_decoder_get_length_in_pcm_frames(&temp_decoder, &total_frames) == MA_SUCCESS) {
                read_len = std::min(static_cast<size_t>(total_frames), max_ir_size);
                ma_decoder_read_pcm_frames(&temp_decoder, ir_raw.data(), read_len, NULL);
            }
            ma_decoder_uninit(&temp_decoder);
        }

        if (read_len > 0) {
            
            if (type == CATHEDRAL) convolverCath.init(1024, ir_raw.data(), read_len);
            else convolverCorr.init(1024, ir_raw.data(), read_len);
        }
    }
};