# frost-fft
The world didn't need yet another Fast Fourier Transform (FFT) implementation, but here we are...

```typescript
import {fft, ifft, ifftReal} from 'frost-fft';

const signal = new Float64Array(256).map(Math.random);

// The imaginary argument is optional, zeros assumed by default (faster).
const [realCoefs, imagCoefs] = fft(
  signal,
  signal.map(() => 0)
);

// There's no normalization. These are 256 times too large.
const [realSignalScaled, imagSignalScaled] = ifft(realCoefs, imagCoefs);

// The original signal reconstructed (with some floating point noise).
const realSignal = ifftReal(realCoefs, imagCoefs).map(s => s / 256);
```

## Documentation ##
Documentation is hosted at the project [Github pages](https://frostburn.github.io/frost-fft).

To generate documentation locally run:
```bash
npm run doc
```
