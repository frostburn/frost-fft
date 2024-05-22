import {describe, it, expect} from 'vitest';
import {ceilPow2, fft, ifft, ifftReal} from '..';

function ft(realIn: Float64Array, imagIn: Float64Array) {
  const N = realIn.length;
  const realOut = new Float64Array(N);
  const imagOut = new Float64Array(N);
  for (let i = 0; i < N; ++i) {
    for (let j = 0; j < N; ++j) {
      const theta = (-2 * Math.PI * i * j) / N;
      const realZ = Math.cos(theta);
      const imagZ = Math.sin(theta);
      realOut[i] += realZ * realIn[j] - imagZ * imagIn[j];
      imagOut[i] += realZ * imagIn[j] + imagZ * realIn[j];
    }
  }
  return [realOut, imagOut];
}

function ift(realIn: Float64Array, imagIn: Float64Array) {
  const N = realIn.length;
  const realOut = new Float64Array(N);
  const imagOut = new Float64Array(N);
  for (let i = 0; i < N; ++i) {
    for (let j = 0; j < N; ++j) {
      const theta = (2 * Math.PI * i * j) / N;
      const realZ = Math.cos(theta);
      const imagZ = Math.sin(theta);
      realOut[i] += realZ * realIn[j] - imagZ * imagIn[j];
      imagOut[i] += realZ * imagIn[j] + imagZ * realIn[j];
    }
  }
  return [realOut, imagOut];
}

describe('Fast Fourier transform', () => {
  it('agrees with the naive implementation in the N = 1, 2, 4, 8 base cases', () => {
    for (let N = 1; N <= 8; N <<= 1) {
      for (let i = 0; i < N; ++i) {
        const realIn = new Float64Array(N);
        const imagIn = new Float64Array(N);
        realIn[i] = 1;
        {
          const [realNaive, imagNaive] = ft(realIn, imagIn);
          const [realCoefs, imagCoefs] = fft(realIn, imagIn);
          const [realCoefs1, imagCoefs1] = fft(realIn);
          for (let j = 0; j < N; ++j) {
            expect(realCoefs[j], `real #${i} -> real #${j}`).toBeCloseTo(
              realNaive[j]
            );
            expect(imagCoefs[j], `real #${i} -> imag #${j}`).toBeCloseTo(
              imagNaive[j]
            );
            expect(realCoefs1[j], `real #${i} -> real #${j}`).toBeCloseTo(
              realNaive[j]
            );
            expect(imagCoefs1[j], `real #${i} -> imag #${j}`).toBeCloseTo(
              imagNaive[j]
            );
          }
        }
        realIn[i] = 0;
        imagIn[i] = 1;
        const [realNaive, imagNaive] = ft(realIn, imagIn);
        const [realCoefs, imagCoefs] = fft(realIn, imagIn);
        for (let j = 0; j < N; ++j) {
          expect(realCoefs[j], `imag #${i} -> real #${j}`).toBeCloseTo(
            realNaive[j]
          );
          expect(imagCoefs[j], `imag #${i} -> imag #${j}`).toBeCloseTo(
            imagNaive[j]
          );
        }
      }
    }
  });

  it('calculates the coefficients for a cosine', () => {
    const N = 16;
    const signal = new Float64Array(N).map((_, k) =>
      Math.cos((2 * Math.PI * k) / N)
    );
    const [realCoefs, imagCoefs] = fft(
      signal,
      signal.map(() => 0)
    );
    const [realCoefs1, imagCoefs1] = fft(signal);

    expect(realCoefs[1]).toBeCloseTo(N / 2);
    expect(realCoefs1[1]).toBeCloseTo(N / 2);
    expect(realCoefs[N - 1]).toBeCloseTo(N / 2);
    expect(realCoefs1[N - 1]).toBeCloseTo(N / 2);

    realCoefs[1] = 0;
    realCoefs1[1] = 0;
    realCoefs[N - 1] = 0;
    realCoefs1[N - 1] = 0;

    for (let i = 0; i < N; ++i) {
      expect(realCoefs[i]).toBeCloseTo(0);
      expect(realCoefs1[i]).toBeCloseTo(0);
      expect(imagCoefs[i]).toBeCloseTo(0);
      expect(imagCoefs1[i]).toBeCloseTo(0);
    }
  });

  it('throws if the input is not a power of two', () => {
    const N = 5;
    const real = new Float64Array(N);
    const imag = new Float64Array(N);
    expect(() => fft(real, imag)).toThrow();
    expect(() => fft(real)).toThrow();
  });

  it('throws if the input lengths do not match', () => {
    const real = new Float64Array(4);
    const imag = new Float64Array(8);
    expect(() => fft(real, imag)).toThrow();
  });
});

describe('Inverse fast Fourier transform', () => {
  it('agrees with the naive implementation in the N = 8 base case', () => {
    const N = 8;
    for (let i = 0; i < N; ++i) {
      const realIn = new Float64Array(N);
      const imagIn = new Float64Array(N);
      realIn[i] = 1;
      {
        const [realNaive, imagNaive] = ift(realIn, imagIn);
        const [realCoefs, imagCoefs] = ifft(realIn, imagIn);
        const realCoefs1 = ifftReal(realIn, imagIn);
        for (let j = 0; j < N; ++j) {
          expect(realCoefs[j], `real #${i} -> real #${j}`).toBeCloseTo(
            realNaive[j]
          );
          expect(imagCoefs[j], `real #${i} -> imag #${j}`).toBeCloseTo(
            imagNaive[j]
          );
          expect(realCoefs1[j], `real #${i} -> real #${j}`).toBeCloseTo(
            realNaive[j]
          );
        }
      }
      realIn[i] = 0;
      imagIn[i] = 1;
      const [realNaive, imagNaive] = ift(realIn, imagIn);
      const [realCoefs, imagCoefs] = ifft(realIn, imagIn);
      const realCoefs1 = ifftReal(realIn, imagIn);
      for (let j = 0; j < N; ++j) {
        expect(realCoefs[j], `imag #${i} -> real #${j}`).toBeCloseTo(
          realNaive[j]
        );
        expect(imagCoefs[j], `imag #${i} -> imag #${j}`).toBeCloseTo(
          imagNaive[j]
        );
        expect(realCoefs1[j], `imag #${i} -> real #${j}`).toBeCloseTo(
          realNaive[j]
        );
      }
    }
  });

  it('creates a cosine', () => {
    const N = 16;
    const real = new Float64Array(N);
    const imag = new Float64Array(N);
    real[1] = 0.5;
    real[N - 1] = 0.5;

    const [realOut, imagOut] = ifft(real, imag);
    const realOut1 = ifftReal(real, imag);

    for (let i = 0; i < N; ++i) {
      const theta = (2 * Math.PI * i) / N;
      expect(realOut[i]).toBeCloseTo(Math.cos(theta));
      expect(imagOut[i]).toBeCloseTo(0);
      expect(realOut1[i]).toBeCloseTo(Math.cos(theta));
    }
  });

  it('agrees with the naive implementation', () => {
    const N = 32;
    const real = new Float64Array(N).map(Math.random);
    const imag = new Float64Array(N).map(Math.random);

    const [realOut, imagOut] = ifft(real, imag);
    const realOut1 = ifftReal(real, imag);
    const [realReference, imagReference] = ift(real, imag);

    for (let i = 0; i < N; ++i) {
      expect(realOut[i]).toBeCloseTo(realReference[i]);
      expect(imagOut[i]).toBeCloseTo(imagReference[i]);
      expect(realOut1[i]).toBeCloseTo(realReference[i]);
    }
  });

  it('undoes FFT with the appropriate normalization', () => {
    const N = 64;
    const real = new Float64Array(N).map(Math.random);
    const imag = new Float64Array(N);

    const [realCoefs, imagCoefs] = fft(real, imag);
    const [realCoefs1, imagCoefs1] = fft(real);
    const [realOut, imagOut] = ifft(realCoefs, imagCoefs);
    const realOut1 = ifftReal(realCoefs1, imagCoefs1);

    for (let i = 0; i < N; ++i) {
      expect(realOut[i]).toBeCloseTo(real[i] * N);
      expect(imagOut[i]).toBeCloseTo(imag[i] * N);
      expect(realOut1[i]).toBeCloseTo(real[i] * N);
    }
  });

  it('throws if the input is not a power of two', () => {
    const N = 3;
    const real = new Float64Array(N);
    const imag = new Float64Array(N);
    expect(() => ifft(real, imag)).toThrow();
    expect(() => ifftReal(real, imag)).toThrow();
  });

  it('throws if the input lengths do not match', () => {
    const real = new Float64Array(4);
    const imag = new Float64Array(2);
    expect(() => ifft(real, imag)).toThrow();
    expect(() => ifftReal(real, imag)).toThrow();
  });
});

describe('Upper power of two', () => {
  it.each([0, 1, 2, 3, 4, 5, 6])('matches 2**%s', n => {
    expect(ceilPow2(1 << n)).toBe(1 << n);
  });

  it('is not less than than the input value', () => {
    const value = Math.floor(Math.random() * 2 ** 30);
    expect(ceilPow2(value)).not.toBeLessThan(value);
  });
});
