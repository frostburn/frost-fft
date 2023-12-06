# frost-fft
The world didn't need yet another Fast Fourier Transform (FFT) implementation, but here we are...

```typescript
import {fft} from 'frost-fft';

const signal = new Float64Array(256).map(Math.random);

const [realCoefs, imagCoefs] = fft(
  signal,
  signal.map(() => 0)
);
```
