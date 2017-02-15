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


class MFT():
    """
    Encapsulates the computation of the MFT
    """

    def __init__(self, series, sampling):
        """

        :param series: a numpy array with the data (real values)
        :param sampling: sampling of the signal
        """
        self.series = series
        self.samplig = sampling


    def compute(self, ncoef, wsize):
        """
        Computes the nfreq fourier coefficient for the series
        :return:
        """

        nwindows = len(self.series)-wsize
        # imaginary matrix for the coefficients
        coef = np.zeros((nwindows, ncoef), dtype=np.complex)

        y = np.fft.rfft(self.series[:wsize])
        for l in range(ncoef):
            coef[l, 0] = y[l]

        for w in range(1, nwindows):
            for l in range(ncoef):
                fk = np.exp(1j*2*np.pi*((l)/float(wsize)))

                yk = fk * (y[l] + (self.series[wsize+w]-self.series[w]))

                coef[l,w] = yk
                y[l] = coef[l,w]



