const TAU = 2 * Math.PI;

const COSINES: number[][] = [];
const SINES: number[][] = [];

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
 * @param imagIn Imaginary components of the signal.
 * @returns Array of [real coefficients, imaginary coefficients].
 */
export function fft(realIn: Float64Array, imagIn: Float64Array) {
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

function _fft(realIn: Float64Array, imagIn: Float64Array) {
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

  const cosines = COSINES[M] ?? [];
  const sines = SINES[M] ?? [];
  if (cosines.length) {
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
  } else {
    const TAU_OVER_N = TAU / N;
    for (let k = 1; k < M; ++k) {
      const realZ = (cosines[k] = Math.cos(k * TAU_OVER_N));
      const imagZ = -(sines[k] = Math.sin(k * TAU_OVER_N));
      const realQ = realOdds[k] * realZ - imagOdds[k] * imagZ;
      const imagQ = realOdds[k] * imagZ + imagOdds[k] * realZ;
      realOut[k] = realEvens[k] + realQ;
      imagOut[k] = imagEvens[k] + imagQ;

      realOut[k + M] = realEvens[k] - realQ;
      imagOut[k + M] = imagEvens[k] - imagQ;
    }
    COSINES[M] = cosines;
    SINES[M] = sines;
  }

  return [realOut, imagOut];
}

/**
 * Calculate the unnormalized reverse discrete Fourier transform.
 * @param realIn Real coefficients of a forward transform.
 * @param imagIn Imaginary coefficients of a forward transform.
 * @returns Array of [real signal, imaginary signal] scaled by the length of the original signal.
 */
export function ifft(realIn: Float64Array, imagIn: Float64Array) {
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

function _ifft(realIn: Float64Array, imagIn: Float64Array) {
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

  const cosines = COSINES[M] ?? [];
  const sines = SINES[M] ?? [];
  if (cosines.length) {
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
  } else {
    const TAU_OVER_N = TAU / N;
    for (let k = 1; k < M; ++k) {
      const realZ = (cosines[k] = Math.cos(k * TAU_OVER_N));
      const imagZ = (sines[k] = Math.sin(k * TAU_OVER_N));
      const realQ = realOdds[k] * realZ - imagOdds[k] * imagZ;
      const imagQ = realOdds[k] * imagZ + imagOdds[k] * realZ;
      realOut[k] = realEvens[k] + realQ;
      imagOut[k] = imagEvens[k] + imagQ;

      realOut[k + M] = realEvens[k] - realQ;
      imagOut[k + M] = imagEvens[k] - imagQ;
    }
    COSINES[M] = cosines;
    SINES[M] = sines;
  }

  return [realOut, imagOut];
}
