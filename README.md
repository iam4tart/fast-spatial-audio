# Fast Spatial Audio

Experiment: map 3D space → Morton (1D memory) → drive simple real-time audio response.

Context: https://4t-audio.vercel.app/blog/fast-spatial-audio



## Where to look

- `bb_spatial.h`  
  quantization + Morton encoding (world → grid → 1D)

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

- Is Morton layout useful here or unnecessary?
- Is IR blending a reasonable spatial model?
- Better way to couple spatial structure → audio?
- Any obvious issues in DSP / threading?



Build:

```bash
mkdir build
cmake -B build && cmake --build build
.\build\Debug\app.exe
```