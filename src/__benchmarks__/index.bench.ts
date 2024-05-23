import {describe, bench, beforeAll} from 'vitest';
import {fft, ifft, ifftReal} from '../';

describe('Empty 1024', () => {
  beforeAll(() => {
    const real = new Float64Array(1024);
    const imag = new Float64Array(1024);
    fft(real, imag);
    ifft(real, imag);
  });

  bench('Forward', () => {
    const real = new Float64Array(1024);
    const imag = new Float64Array(1024);
    fft(real, imag);
  });

  bench('Inverse', () => {
    const real = new Float64Array(1024);
    const imag = new Float64Array(1024);
    ifft(real, imag);
  });

  bench('Forward no imaginary', () => {
    const real = new Float64Array(1024);
    fft(real);
  });

  bench('Inverse real result', () => {
    const real = new Float64Array(1024);
    const imag = new Float64Array(1024);
    ifftReal(real, imag);
  });
});

function randomData(): [Float64Array, Float64Array] {
  const N = Math.floor(11 + Math.random() * 4);
  const real = new Float64Array(1 << N);
  const imag = new Float64Array(1 << N);
  for (let i = 0; i < N; ++i) {
    real[i] = Math.random() * 2 - 1;
    imag[i] = Math.random() * 2 - 1;
  }
  return [real, imag];
}

describe('Random power of two', () => {
  beforeAll(() => {
    for (let i = 1; i < 16; ++i) {
      const real = new Float64Array(1 << i);
      const imag = new Float64Array(1 << i);
      fft(real, imag);
      ifft(real, imag);
    }
  });

  bench('Forward', () => {
    fft(...randomData());
  });

  bench('Inverse', () => {
    ifft(...randomData());
  });

  bench('Forward no imaginary', () => {
    const N = Math.floor(11 + Math.random() * 4);
    const real = new Float64Array(1 << N);
    for (let i = 0; i < N; ++i) {
      real[i] = Math.random() * 2 - 1;
    }
    fft(real);
  });

  bench('Inverse real result', () => {
    ifftReal(...randomData());
  });
});
