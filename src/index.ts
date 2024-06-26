const COSINES: number[][] = [];
const SINES: number[][] = [];

COSINES[2] = [NaN, 0];
SINES[2] = [NaN, 1];

/**
 * Reset internal tables. Used during benchmarking.
 */
export function _resetTables() {
  COSINES.length = 0;
  SINES.length = 0;
  COSINES[2] = [NaN, 0];
  SINES[2] = [NaN, 1];
}

function getTables(M: number): [number[], number[]] {
  const cosines = COSINES[M] ?? [];
  const sines = SINES[M] ?? [];
  if (cosines.length) {
    return [cosines, sines];
  }
  cosines.length = M;
  sines.length = M;
  const L = M >>> 1;
  const [c, s] = getTables(L);
  // i = 0 intentionally left undefined
  for (let i = 1; i < L; ++i) {
    cosines[i << 1] = c[i];
    sines[i << 1] = s[i];
  }
  const PI_OVER_M = Math.PI / M;
  for (let i = 1; i < L; i += 2) {
    sines[i] = sines[M - i] = cosines[L - i] = Math.sin(i * PI_OVER_M);
    cosines[L + i] = -cosines[L - i];
  }
  return [cosines, sines];
}

/**
 * Calculate the smallest power of two greater or equal to the input value.
 * @param x Integer to compare to.
 * @returns Smallest `2**n` such that `x <= 2**n`.
 */
export function ceilPow2(x: number) {
  return 1 << (32 - Math.clz32(x - 1));
}

/**
 * Calculate the unnormalized forward discrete Fourier transform.
 * @param realIn Real components of the signal.
 * @param imagIn Imaginary components of the signal (all zeros assumed if missing).
 * @returns Array of [real coefficients, imaginary coefficients].
 */
export function fft(
  realIn: Float64Array,
  imagIn?: Float64Array
): [Float64Array, Float64Array] {
  const N = realIn.length;
  if (N !== ceilPow2(N)) {
    throw new Error('Length must be a power of two.');
  }
  if (imagIn === undefined) {
    return _fftNoImag(realIn);
  }
  if (imagIn.length !== N) {
    throw new Error(
      'Must have an equal number of real and imaginary components'
    );
  }

  const realOut = new Float64Array(N);
  const imagOut = new Float64Array(N);
  if (N === 4) {
    realOut[0] = realIn[0] + realIn[1] + realIn[2] + realIn[3];
    realOut[1] = realIn[0] + imagIn[1] - realIn[2] - imagIn[3];
    realOut[2] = realIn[0] - realIn[1] + realIn[2] - realIn[3];
    realOut[3] = realIn[0] - imagIn[1] - realIn[2] + imagIn[3];

    imagOut[0] = imagIn[0] + imagIn[1] + imagIn[2] + imagIn[3];
    imagOut[1] = imagIn[0] - realIn[1] - imagIn[2] + realIn[3];
    imagOut[2] = imagIn[0] - imagIn[1] + imagIn[2] - imagIn[3];
    imagOut[3] = imagIn[0] + realIn[1] - imagIn[2] - realIn[3];
    return [realOut, imagOut];
  }
  if (N === 2) {
    realOut[0] = realIn[0] + realIn[1];
    realOut[1] = realIn[0] - realIn[1];
    imagOut[0] = imagIn[0] + imagIn[1];
    imagOut[1] = imagIn[0] - imagIn[1];
    return [realOut, imagOut];
  }
  if (N === 1) {
    realOut[0] = realIn[0];
    imagOut[0] = imagIn[0];
    return [realOut, imagOut];
  }
  return _fft(realIn, imagIn);
}

function _fft(
  realIn: Float64Array,
  imagIn: Float64Array
): [Float64Array, Float64Array] {
  const N = realIn.length;
  const realOut = new Float64Array(N);
  const imagOut = new Float64Array(N);
  if (N === 8) {
    const realEvens0 = realIn[0] + realIn[2] + realIn[4] + realIn[6];
    const imagEvens0 = imagIn[0] + imagIn[2] + imagIn[4] + imagIn[6];
    const realOdds0 = realIn[1] + realIn[3] + realIn[5] + realIn[7];
    const imagOdds0 = imagIn[1] + imagIn[3] + imagIn[5] + imagIn[7];
    realOut[0] = realEvens0 + realOdds0;
    imagOut[0] = imagEvens0 + imagOdds0;
    realOut[4] = realEvens0 - realOdds0;
    imagOut[4] = imagEvens0 - imagOdds0;

    const realEvens1 = realIn[0] + imagIn[2] - realIn[4] - imagIn[6];
    const imagEvens1 = imagIn[0] - realIn[2] - imagIn[4] + realIn[6];
    const realOdds1 = realIn[1] + imagIn[3] - realIn[5] - imagIn[7];
    const imagOdds1 = imagIn[1] - realIn[3] - imagIn[5] + realIn[7];
    const realQ1 = (realOdds1 + imagOdds1) * 0.7071067811865475;
    const imagQ1 = (realOdds1 - imagOdds1) * 0.7071067811865475;
    realOut[1] = realEvens1 + realQ1;
    imagOut[1] = imagEvens1 - imagQ1;
    realOut[5] = realEvens1 - realQ1;
    imagOut[5] = imagEvens1 + imagQ1;

    const realEvens2 = realIn[0] - realIn[2] + realIn[4] - realIn[6];
    const imagEvens2 = imagIn[0] - imagIn[2] + imagIn[4] - imagIn[6];
    const realOdds2 = realIn[1] - realIn[3] + realIn[5] - realIn[7];
    const imagOdds2 = imagIn[1] - imagIn[3] + imagIn[5] - imagIn[7];
    realOut[2] = realEvens2 + imagOdds2;
    imagOut[2] = imagEvens2 - realOdds2;
    realOut[6] = realEvens2 - imagOdds2;
    imagOut[6] = imagEvens2 + realOdds2;

    const realEvens3 = realIn[0] - imagIn[2] - realIn[4] + imagIn[6];
    const imagEvens3 = imagIn[0] + realIn[2] - imagIn[4] - realIn[6];
    const realOdds3 = realIn[1] - imagIn[3] - realIn[5] + imagIn[7];
    const imagOdds3 = imagIn[1] + realIn[3] - imagIn[5] - realIn[7];
    const realQ3 = (realOdds3 - imagOdds3) * 0.7071067811865475;
    const imagQ3 = (realOdds3 + imagOdds3) * 0.7071067811865475;
    realOut[3] = realEvens3 - realQ3;
    imagOut[3] = imagEvens3 - imagQ3;
    realOut[7] = realEvens3 + realQ3;
    imagOut[7] = imagEvens3 + imagQ3;

    return [realOut, imagOut];
  }

  const [realEvens, imagEvens] = _fft(
    realIn.filter((_, k) => !(k & 1)),
    imagIn.filter((_, k) => !(k & 1))
  );
  const [realOdds, imagOdds] = _fft(
    realIn.filter((_, k) => k & 1),
    imagIn.filter((_, k) => k & 1)
  );

  const M = N >>> 1;

  realOut[0] = realEvens[0] + realOdds[0];
  realOut[M] = realEvens[0] - realOdds[0];
  imagOut[0] = imagEvens[0] + imagOdds[0];
  imagOut[M] = imagEvens[0] - imagOdds[0];

  const [cosines, sines] = getTables(M);
  for (let k = 1; k < M; ++k) {
    const realZ = cosines[k];
    const imagZ = -sines[k];
    const realQ = realOdds[k] * realZ - imagOdds[k] * imagZ;
    const imagQ = realOdds[k] * imagZ + imagOdds[k] * realZ;
    realOut[k] = realEvens[k] + realQ;
    imagOut[k] = imagEvens[k] + imagQ;

    realOut[k + M] = realEvens[k] - realQ;
    imagOut[k + M] = imagEvens[k] - imagQ;
  }

  return [realOut, imagOut];
}

function _fftNoImag(realIn: Float64Array): [Float64Array, Float64Array] {
  const N = realIn.length;
  const realOut = new Float64Array(N);
  const imagOut = new Float64Array(N);
  if (N === 4) {
    realOut[0] = realIn[0] + realIn[1] + realIn[2] + realIn[3];
    realOut[1] = realIn[0] - realIn[2];
    realOut[2] = realIn[0] - realIn[1] + realIn[2] - realIn[3];
    realOut[3] = realIn[0] - realIn[2];

    imagOut[1] = realIn[3] - realIn[1];
    imagOut[3] = realIn[1] - realIn[3];
    return [realOut, imagOut];
  }
  if (N === 2) {
    realOut[0] = realIn[0] + realIn[1];
    realOut[1] = realIn[0] - realIn[1];
    return [realOut, imagOut];
  }
  if (N === 1) {
    realOut[0] = realIn[0];
    return [realOut, imagOut];
  }
  return _fftNoImagInner(realIn);
}

function _fftNoImagInner(realIn: Float64Array): [Float64Array, Float64Array] {
  const N = realIn.length;
  const realOut = new Float64Array(N);
  const imagOut = new Float64Array(N);
  if (N === 8) {
    const realEvens0 = realIn[0] + realIn[2] + realIn[4] + realIn[6];
    const realOdds0 = realIn[1] + realIn[3] + realIn[5] + realIn[7];
    realOut[0] = realEvens0 + realOdds0;
    realOut[4] = realEvens0 - realOdds0;

    const realEvens1 = realIn[0] - realIn[4];
    const imagEvens1 = realIn[6] - realIn[2];
    const realOdds1 = realIn[1] - realIn[5];
    const imagOdds1 = realIn[7] - realIn[3];
    const realQ1 = (realOdds1 + imagOdds1) * 0.7071067811865475;
    const imagQ1 = (realOdds1 - imagOdds1) * 0.7071067811865475;
    realOut[1] = realEvens1 + realQ1;
    imagOut[1] = imagEvens1 - imagQ1;
    realOut[5] = realEvens1 - realQ1;
    imagOut[5] = imagEvens1 + imagQ1;

    const realEvens2 = realIn[0] - realIn[2] + realIn[4] - realIn[6];
    const realOdds2 = realIn[1] - realIn[3] + realIn[5] - realIn[7];
    realOut[2] = realEvens2;
    imagOut[2] = -realOdds2;
    realOut[6] = realEvens2;
    imagOut[6] = realOdds2;

    const realEvens3 = realIn[0] - realIn[4];
    const imagEvens3 = realIn[2] - realIn[6];
    const realOdds3 = realIn[1] - realIn[5];
    const imagOdds3 = realIn[3] - realIn[7];
    const realQ3 = (realOdds3 - imagOdds3) * 0.7071067811865475;
    const imagQ3 = (realOdds3 + imagOdds3) * 0.7071067811865475;
    realOut[3] = realEvens3 - realQ3;
    imagOut[3] = imagEvens3 - imagQ3;
    realOut[7] = realEvens3 + realQ3;
    imagOut[7] = imagEvens3 + imagQ3;

    return [realOut, imagOut];
  }

  const [realEvens, imagEvens] = _fftNoImagInner(
    realIn.filter((_, k) => !(k & 1))
  );
  const [realOdds, imagOdds] = _fftNoImagInner(realIn.filter((_, k) => k & 1));

  const M = N >>> 1;

  realOut[0] = realEvens[0] + realOdds[0];
  realOut[M] = realEvens[0] - realOdds[0];
  imagOut[0] = imagEvens[0] + imagOdds[0];
  imagOut[M] = imagEvens[0] - imagOdds[0];

  const [cosines, sines] = getTables(M);
  for (let k = 1; k < M; ++k) {
    const realZ = cosines[k];
    const imagZ = -sines[k];
    const realQ = realOdds[k] * realZ - imagOdds[k] * imagZ;
    const imagQ = realOdds[k] * imagZ + imagOdds[k] * realZ;
    realOut[k] = realEvens[k] + realQ;
    imagOut[k] = imagEvens[k] + imagQ;

    realOut[k + M] = realEvens[k] - realQ;
    imagOut[k + M] = imagEvens[k] - imagQ;
  }

  return [realOut, imagOut];
}

/**
 * Calculate the unnormalized reverse discrete Fourier transform.
 * @param realIn Real coefficients of a forward transform.
 * @param imagIn Imaginary coefficients of a forward transform.
 * @returns Array of [real signal, imaginary signal] scaled by the length of the original signal.
 */
export function ifft(
  realIn: Float64Array,
  imagIn: Float64Array
): [Float64Array, Float64Array] {
  const N = realIn.length;
  if (N !== ceilPow2(N)) {
    throw new Error('Length must be a power of two.');
  }
  if (imagIn.length !== N) {
    throw new Error(
      'Must have an equal number of real and imaginary components'
    );
  }
  const realOut = new Float64Array(N);
  const imagOut = new Float64Array(N);
  if (N === 4) {
    realOut[0] = realIn[0] + realIn[1] + realIn[2] + realIn[3];
    realOut[1] = realIn[0] - imagIn[1] - realIn[2] + imagIn[3];
    realOut[2] = realIn[0] - realIn[1] + realIn[2] - realIn[3];
    realOut[3] = realIn[0] + imagIn[1] - realIn[2] - imagIn[3];

    imagOut[0] = imagIn[0] + imagIn[1] + imagIn[2] + imagIn[3];
    imagOut[1] = imagIn[0] + realIn[1] - imagIn[2] - realIn[3];
    imagOut[2] = imagIn[0] - imagIn[1] + imagIn[2] - imagIn[3];
    imagOut[3] = imagIn[0] - realIn[1] - imagIn[2] + realIn[3];
    return [realOut, imagOut];
  }
  if (N === 2) {
    realOut[0] = realIn[0] + realIn[1];
    realOut[1] = realIn[0] - realIn[1];
    imagOut[0] = imagIn[0] + imagIn[1];
    imagOut[1] = imagIn[0] - imagIn[1];
    return [realOut, imagOut];
  }
  if (N === 1) {
    realOut[0] = realIn[0];
    imagOut[0] = imagIn[0];
    return [realOut, imagOut];
  }
  return _ifft(realIn, imagIn);
}

function _ifft(
  realIn: Float64Array,
  imagIn: Float64Array
): [Float64Array, Float64Array] {
  const N = realIn.length;
  const realOut = new Float64Array(N);
  const imagOut = new Float64Array(N);
  if (N === 8) {
    const realEvens0 = realIn[0] + realIn[2] + realIn[4] + realIn[6];
    const imagEvens0 = imagIn[0] + imagIn[2] + imagIn[4] + imagIn[6];
    const realOdds0 = realIn[1] + realIn[3] + realIn[5] + realIn[7];
    const imagOdds0 = imagIn[1] + imagIn[3] + imagIn[5] + imagIn[7];
    realOut[0] = realEvens0 + realOdds0;
    imagOut[0] = imagEvens0 + imagOdds0;
    realOut[4] = realEvens0 - realOdds0;
    imagOut[4] = imagEvens0 - imagOdds0;

    const realEvens1 = realIn[0] - imagIn[2] - realIn[4] + imagIn[6];
    const imagEvens1 = imagIn[0] + realIn[2] - imagIn[4] - realIn[6];
    const realOdds1 = realIn[1] - imagIn[3] - realIn[5] + imagIn[7];
    const imagOdds1 = imagIn[1] + realIn[3] - imagIn[5] - realIn[7];
    const realQ1 = (realOdds1 - imagOdds1) * 0.7071067811865475;
    const imagQ1 = (realOdds1 + imagOdds1) * 0.7071067811865475;
    realOut[1] = realEvens1 + realQ1;
    imagOut[1] = imagEvens1 + imagQ1;
    realOut[5] = realEvens1 - realQ1;
    imagOut[5] = imagEvens1 - imagQ1;

    const realEvens2 = realIn[0] - realIn[2] + realIn[4] - realIn[6];
    const imagEvens2 = imagIn[0] - imagIn[2] + imagIn[4] - imagIn[6];
    const realOdds2 = realIn[1] - realIn[3] + realIn[5] - realIn[7];
    const imagOdds2 = imagIn[1] - imagIn[3] + imagIn[5] - imagIn[7];
    realOut[2] = realEvens2 - imagOdds2;
    imagOut[2] = imagEvens2 + realOdds2;
    realOut[6] = realEvens2 + imagOdds2;
    imagOut[6] = imagEvens2 - realOdds2;

    const realEvens3 = realIn[0] + imagIn[2] - realIn[4] - imagIn[6];
    const imagEvens3 = imagIn[0] - realIn[2] - imagIn[4] + realIn[6];
    const realOdds3 = realIn[1] + imagIn[3] - realIn[5] - imagIn[7];
    const imagOdds3 = imagIn[1] - realIn[3] - imagIn[5] + realIn[7];
    const realQ3 = (realOdds3 + imagOdds3) * 0.7071067811865475;
    const imagQ3 = (realOdds3 - imagOdds3) * 0.7071067811865475;
    realOut[3] = realEvens3 - realQ3;
    imagOut[3] = imagEvens3 + imagQ3;
    realOut[7] = realEvens3 + realQ3;
    imagOut[7] = imagEvens3 - imagQ3;

    return [realOut, imagOut];
  }
  const [realEvens, imagEvens] = _ifft(
    realIn.filter((_, k) => !(k & 1)),
    imagIn.filter((_, k) => !(k & 1))
  );
  const [realOdds, imagOdds] = _ifft(
    realIn.filter((_, k) => k & 1),
    imagIn.filter((_, k) => k & 1)
  );

  const M = N >>> 1;

  realOut[0] = realEvens[0] + realOdds[0];
  realOut[M] = realEvens[0] - realOdds[0];
  imagOut[0] = imagEvens[0] + imagOdds[0];
  imagOut[M] = imagEvens[0] - imagOdds[0];

  const [cosines, sines] = getTables(M);

  for (let k = 1; k < M; ++k) {
    const realZ = cosines[k];
    const imagZ = sines[k];
    const realQ = realOdds[k] * realZ - imagOdds[k] * imagZ;
    const imagQ = realOdds[k] * imagZ + imagOdds[k] * realZ;
    realOut[k] = realEvens[k] + realQ;
    imagOut[k] = imagEvens[k] + imagQ;

    realOut[k + M] = realEvens[k] - realQ;
    imagOut[k + M] = imagEvens[k] - imagQ;
  }

  return [realOut, imagOut];
}

/**
 * Calculate the unnormalized reverse discrete Fourier transform.
 * @param realIn Real coefficients of a forward transform.
 * @param imagIn Imaginary coefficients of a forward transform.
 * @returns Real signal scaled by the length of the original signal.
 */
export function ifftReal(
  realIn: Float64Array,
  imagIn: Float64Array
): Float64Array {
  const N = realIn.length;
  if (N !== ceilPow2(N)) {
    throw new Error('Length must be a power of two.');
  }
  if (imagIn.length !== N) {
    throw new Error(
      'Must have an equal number of real and imaginary components'
    );
  }
  const realOut = new Float64Array(N);
  if (N === 4) {
    realOut[0] = realIn[0] + realIn[1] + realIn[2] + realIn[3];
    realOut[1] = realIn[0] - imagIn[1] - realIn[2] + imagIn[3];
    realOut[2] = realIn[0] - realIn[1] + realIn[2] - realIn[3];
    realOut[3] = realIn[0] + imagIn[1] - realIn[2] - imagIn[3];

    return realOut;
  }
  if (N === 2) {
    realOut[0] = realIn[0] + realIn[1];
    realOut[1] = realIn[0] - realIn[1];
    return realOut;
  }
  if (N === 1) {
    realOut[0] = realIn[0];
    return realOut;
  }
  return _ifftReal(realIn, imagIn);
}

function _ifftReal(realIn: Float64Array, imagIn: Float64Array): Float64Array {
  const N = realIn.length;
  const realOut = new Float64Array(N);
  if (N === 8) {
    const realEvens0 = realIn[0] + realIn[2] + realIn[4] + realIn[6];
    const realOdds0 = realIn[1] + realIn[3] + realIn[5] + realIn[7];
    realOut[0] = realEvens0 + realOdds0;
    realOut[4] = realEvens0 - realOdds0;

    const realEvens1 = realIn[0] - imagIn[2] - realIn[4] + imagIn[6];
    const realOdds1 = realIn[1] - imagIn[3] - realIn[5] + imagIn[7];
    const imagOdds1 = imagIn[1] + realIn[3] - imagIn[5] - realIn[7];
    const realQ1 = (realOdds1 - imagOdds1) * 0.7071067811865475;
    realOut[1] = realEvens1 + realQ1;
    realOut[5] = realEvens1 - realQ1;

    const realEvens2 = realIn[0] - realIn[2] + realIn[4] - realIn[6];
    const imagOdds2 = imagIn[1] - imagIn[3] + imagIn[5] - imagIn[7];
    realOut[2] = realEvens2 - imagOdds2;
    realOut[6] = realEvens2 + imagOdds2;

    const realEvens3 = realIn[0] + imagIn[2] - realIn[4] - imagIn[6];
    const realOdds3 = realIn[1] + imagIn[3] - realIn[5] - imagIn[7];
    const imagOdds3 = imagIn[1] - realIn[3] - imagIn[5] + realIn[7];
    const realQ3 = (realOdds3 + imagOdds3) * 0.7071067811865475;
    realOut[3] = realEvens3 - realQ3;
    realOut[7] = realEvens3 + realQ3;

    return realOut;
  }
  const realEvens = _ifftReal(
    realIn.filter((_, k) => !(k & 1)),
    imagIn.filter((_, k) => !(k & 1))
  );
  const [realOdds, imagOdds] = _ifft(
    realIn.filter((_, k) => k & 1),
    imagIn.filter((_, k) => k & 1)
  );

  const M = N >>> 1;

  realOut[0] = realEvens[0] + realOdds[0];
  realOut[M] = realEvens[0] - realOdds[0];

  const [cosines, sines] = getTables(M);

  for (let k = 1; k < M; ++k) {
    const realQ = realOdds[k] * cosines[k] - imagOdds[k] * sines[k];
    realOut[k] = realEvens[k] + realQ;

    realOut[k + M] = realEvens[k] - realQ;
  }

  return realOut;
}
