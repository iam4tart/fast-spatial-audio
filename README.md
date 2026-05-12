# Fast Spatial Audio

Experiment: map 3D space → Morton (1D memory) → drive simple real-time audio response.

Context: [THINKING BLOG](https://4t-audio.vercel.app/blog/fast-spatial-audio)

Interface:

<!-- ![Interface](/images/interface.png) -->
<video src="images/demo.mp4" controls="controls" muted="muted" style="max-width: 100%;"></video>

## Where to look

- `bb_spatial.h`  
  quantization + octree + Morton encoding (world → grid → 1D)

- `audio_engine.h`  
  real-time pipeline (miniaudio + FFT convolution + IR blending)

- `main.cpp`  
  wiring + visualization (listener, zones, params)



## What it does

- Quantizes a 3D room into a grid
- Maps cells to 1D (Morton order)
- Listener movement updates:
  - gain (distance)
  - IR blend (cathedral ↔ corridor)
- Audio processed in real-time with convolution



## Open questions

- Does mapping space with Morton order make sense here, or is it premature?
- Is dual-IR blending a reasonable approximation for spatial transitions?
- Is there a better way to connect spatial structure to audio behavior?
- Any obvious issues in DSP / threading?



## Build

```bash
mkdir build
cmake -B build && cmake --build build
.\build\Debug\app.exe
```

> NOTE: If build fails: system deps (GLFW/OpenGL) may be missing — local setup assumed for now.