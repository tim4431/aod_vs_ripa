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
- I want an animation render. On the left, show atom plane. On the right, show RIPA/AOD frequency tones, and their time trajectories. See [](vis_example.png).
- The animation render generate pictures and save as gif. Use multiprocessing to accelerate gif creating speed. Have a switch to optimize for speed or performance.
- Before the re-arrangement start, and after the re-arrangment ends, stop for 1 second in gif for better animation.


## How to check visualization
- Plot at t=0, when all atoms show their planned trajectories
- Plot at t=final, to see if atoms are in the correct final config
- Generate animation gif to let humans validate if the movement is correct.
- Use suffix to distinguish these figures. Use your multimodal abilities to read these figures and check.