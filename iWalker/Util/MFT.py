"""
.. module:: MFT

MFT
*************

:Description: MFT

    Computes the Momentary Fourier Transformation

    Computes the n firsts coefficients of the FFT for consecutive windows given the n first FFT coeffs of the first window
    (cost is O(nwindows*ncoef) instead of  O(wsize*log(wsize)*nwindows)

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
import time

__author__ = 'bejar'


def mft(series, sampling, ncoef, wsize, butfirst=False):
    """
    Computes the ncoef fourier coefficient for the series

    butfirst eliminates the first coefficient (mean of the signal)

    :return:
    """

    if butfirst:
        ncoef += 1
    nwindows = len(series) - wsize
    # imaginary matrix for the coefficients
    coef = np.zeros((nwindows, ncoef), dtype=np.complex)
    dcoef = np.zeros(ncoef, dtype=np.complex)
    fwsize = float(wsize)
    pi2i = 1j * 2 * np.pi

    y = np.fft.rfft(series[:wsize])
    for l in range(ncoef):
        coef[0, l] = y[l]
        dcoef[l] = np.exp(pi2i * (l / fwsize))

    for w in range(1, nwindows):
        coef[w] = dcoef * (coef[w - 1] + (series[wsize + (w - 1)] - series[w - 1]))

    if butfirst:
        return coef[:, 1:]
    else:
        return coef


# def compute2(series, sampling, ncoef, wsize):
#         """
#         FFT the usual way for testing purposes
#         :param ncoef:
#         :param wsize:
#         :return:
#         """
#         nwindows = len(series)-wsize
#         # imaginary matrix for the coefficients
#         coef = np.zeros((nwindows, ncoef), dtype=np.complex)
#
#         for w in range(nwindows):
#             y = np.fft.rfft(series[w:w+wsize])
#             for l in range(ncoef):
#                 coef[w, l] = y[l]
#
#
#         return coef

if __name__ == '__main__':
    from iWalker.Data import User, Exercise, Exercises, Pacientes, Trajectory
    from iWalker.Util.Misc import show_list_signals

    p = Pacientes()
    e = Exercises()
    p.from_db(pilot='NOGA')
    e.from_db(pilot='NOGA')
    e.delete_patients(['FSL30'])

    ex = e.iterator().__next__()
    signal = ex.get_forces()[:, 0]
    # show_list_signals([signal])

    print(signal.shape)
    itime = time.time()
    coef1 = mft(signal, sampling=10, ncoef=15, wsize=32)
    ftime = time.time()
    print(ftime - itime)

    # itime = time.time()
    # coef2 = compute2(signal, sampling=10, ncoef=15, wsize=32)
    # ftime = time.time()
    # print(ftime - itime)

    # for i in range(coef1.shape[0]):
    #     print(coef1[i]-coef2[i])
