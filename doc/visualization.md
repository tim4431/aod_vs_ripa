## Visualization design

- White background
- Show N*N grid with dot array.
- label axis with their actual physical dimension. The center of grid is (x=0um,y=0um). This is different from our labeling of atoms (i,j)=(0,0)...(N-1,N-1).
- Show atoms. They could have colors.
- When atom is picked up by RIPA/AOD tweezers, show a red gaussian blob with it to show it is being addressed by the tweezer. Or you can do just an edge color for atoms to show they are in tweezer. Make a switch, i think the edge color is more performance-friendly.
- You should also consider rendering static traps (like from SLMs) in another color (like blue).
- When atom is moving, add a "motion-blur" trail effect to show its speed.
- Add a switch to the visualizer to show its planned trajectories after the current time t.
- The visualizer for showing atoms, eom tones, should all be modularized (pass in a ax params, just plot on ax).
- Visualization has three views: `demo` (one atom-motion plot), `benchmark` (one row of atom-motion plots for several schedulers), and `detail` (atom plane plus RIPA/AOD frequency tones and trajectories).
- The animation render generate pictures and save as gif. Use multiprocessing to accelerate gif creating speed. The `quality` switch is `speed` or `quality`; speed skips Gaussian blobs and motion blur, quality uses higher DPI and motion trails.
- GIF moving-frame count is proportional to the total rearrangement time by default, using one rendered frame every `frame_dt` seconds. Pass an explicit `n_frames` only when a fixed sample count is wanted.
- Before the re-arrangement start, and after the re-arrangment ends, stop for 1 second in gif for better animation.


## How to check visualization
- Plot at t=0, when all atoms show their planned trajectories
- Plot at t=final, to see if atoms are in the correct final config
- Generate animation gif to let humans validate if the movement is correct.
- Use suffix to distinguish these figures. Use your multimodal abilities to read these figures and check.
