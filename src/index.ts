const TAU = 2 * Math.PI;

const COSINES: number[][] = [];
const SINES: number[][] = [];

/**
 * Calculate the unnormalized forward discrete Fourier transform.
 * @param realIn Real components of the signal.
 * @param imagIn Imaginary components of the signal.
 * @returns Array of [real coefficients, imaginary coefficients].
 */
export function fft(realIn: Float64Array, imagIn: Float64Array) {
  const N = realIn.length;
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
  const [realEvens, imagEvens] = fft(
    realIn.filter((_, k) => !(k & 1)),
    imagIn.filter((_, k) => !(k & 1))
  );
  const [realOdds, imagOdds] = fft(
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
  const [realEvens, imagEvens] = ifft(
    realIn.filter((_, k) => !(k & 1)),
    imagIn.filter((_, k) => !(k & 1))
  );
  const [realOdds, imagOdds] = ifft(
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
