## RIPA position-frequency map

RIPAs is 2D spectrometer that maps frequency to position, here we mostly care about its position to frequency mapping. Its free-spectral-range (FSR) is an important parameter. Here we consider a RIPA with `FSR1=3GHz` and `FSR2 = FSR1/N`.

If i ramp the frequency you can see the tweezer continuously move along `x` (row channel, see [here](aod_vs_ripa.md)), and if i rotate it by 90 degree, the tweezer continuously move along `y` (col channel).

- **Row channel** (un-rotated): along `x`, so the corresponding frequency tone is
  ```
  ν_row(i, j) = (j + i / N) · FSR2     (mod FSR1)
  ```
  Used while moving an atom along `x`.

- **Col channel** (90° rotated): along `y`, so the corresponding frequency tone is
  ```
  ν_col(i, j) = (i / N + j) · FSR2     (mod FSR1)
  ```
  Used while moving along `y`.

