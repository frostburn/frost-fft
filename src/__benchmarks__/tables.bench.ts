import {describe, bench} from 'vitest';
import {_resetTables, fft, ifft, ifftReal} from '../';

describe('Empty 1024 (no cache)', () => {
  bench('Forward', () => {
    _resetTables();
    const real = new Float64Array(1024);
    const imag = new Float64Array(1024);
    fft(real, imag);
  });

  bench('Inverse', () => {
    _resetTables();
    const real = new Float64Array(1024);
    const imag = new Float64Array(1024);
    ifft(real, imag);
  });

  bench('Forward no imaginary', () => {
    _resetTables();
    const real = new Float64Array(1024);
    fft(real);
  });

  bench('Inverse real result', () => {
    _resetTables();
    const real = new Float64Array(1024);
    const imag = new Float64Array(1024);
    ifftReal(real, imag);
  });
});
