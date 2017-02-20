"""
.. module:: MFT

MFT
*************

:Description: MFT

    Computes the Momentary Fourier Transformation

    Albrecht, Cumming, Dudas "The Momentary Fourier Transformation Derived from Recursive Matrix Transformation"

@inproceedings{albrecht1997momentary,
  title={The momentary fourier transformation derived from recursive matrix transformations},
  author={Albrecht, S and Cumming, I and Dudas, J},
  booktitle={Digital Signal Processing Proceedings, 1997. DSP 97., 1997 13th International Conference on},
  volume={1},
  pages={337--340},
  year={1997},
  organization={IEEE}
}

:Authors: bejar
    

:Version: 

:Created on: 15/02/2017 13:48 

"""
import numpy as np


__author__ = 'bejar'


def mft(series, sampling, ncoef, wsize):
    """
    Computes the ncoef fourier coefficient for the series
    :return:
    """

    nwindows = len(series) - wsize
    # imaginary matrix for the coefficients
    coef = np.zeros((nwindows, ncoef), dtype=np.complex)

    y = np.fft.rfft(series[:wsize])
    for l in range(ncoef):
        coef[0, l] = y[l]

    for w in range(1, nwindows):
        for l in range(ncoef):
            fk = np.exp(1j * 2 * np.pi * ((l) / float(wsize)))

            yk = fk * (coef[w - 1, l] + (series[wsize + (w - 1)] - series[w - 1]))

            coef[w, l] = yk

    return coef

def compute2(series, sampling, ncoef, wsize):
        """
        FFT the usual way for testing purposes
        :param ncoef:
        :param wsize:
        :return:
        """
        nwindows = len(series)-wsize
        # imaginary matrix for the coefficients
        coef = np.zeros((nwindows, ncoef), dtype=np.complex)

        for w in range(nwindows):
            y = np.fft.rfft(series[w:w+wsize])
            for l in range(ncoef):
                coef[w, l] = y[l]


        return coef

if __name__ == '__main__':
    from iWalker.Data import User, Exercise, Exercises, Pacientes, Trajectory
    from iWalker.Util.Misc import show_list_signals
    p = Pacientes()
    e = Exercises()
    p.from_db(pilot='NOGA')
    e.from_db(pilot='NOGA')
    e.delete_patients(['FSL30'])

    ex = e.iterator().__next__()
    signal = ex.get_forces()[:,0]
    # show_list_signals([signal])

    coef1 = mft(signal, sampling=10, ncoef=2, wsize=20)
    coef2 = compute2(signal, sampling=10, ncoef=2, wsize=20)

    for i in range(coef1.shape[0]):
        print(coef1[i], coef2[i])


