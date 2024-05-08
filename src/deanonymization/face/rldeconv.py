from .abstract import AbstractFaceDeanonymization

import cv2
import numpy as np
from scipy.signal import convolve
import operator


class RldeconvDeanonymization(AbstractFaceDeanonymization):
    """De-Anonymize using Richard-Lucy deconvolution

    This does not use the training data but rather estimates parameters blindly for each image.
    See comments below for the papers used.
    The implementation uses a not-yet-merged pull request in skimage.

    Required pips:
        - opencv2
        - skimage
        - numpy

    Parameters:
        none
    """

    name = "rldeconv"

    def deanonymize(self, image):
        img = cv2.imread(image.get_path())
        filtered = self.deconv(img, None)
        cv2.imwrite(image.get_path(), filtered)

    def deconv(self, img, psf=None, iter=50):
        b, g, r = cv2.split(img / 255.0)

        f_b = self.richardson_lucy(b, psf=psf, iterations=iter, refine_psf=True)
        f_g = self.richardson_lucy(g, psf=psf, iterations=iter, refine_psf=True)
        f_r = self.richardson_lucy(r, psf=psf, iterations=iter, refine_psf=True)

        filtered = cv2.merge([f_b, f_g, f_r])
        filtered *= 255.0 / filtered.max()
        return filtered

    def _pad_to_shape(self, image, psf):
        """
         Will add pad around psf to get same shape as image in blind RL deconvolve
         Parameters
         ----------
         image : ndarray
            Input degraded image (can be N dimensional).
         psf : ndarray
            The point spread function.
        Returns
        -------
        container : ndarray
           Reshaped psf.
        """
        container = np.zeros((image.shape), image.dtype)
        start = tuple(map(lambda a, da: a // 2 - da // 2, container.shape, psf.shape))
        end = tuple(map(operator.add, start, psf.shape))
        slices = tuple(map(slice, start, end))
        container[slices] = psf
        return container

    def richardson_lucy(self, image, psf=None, iterations=50, clip=True, return_iterations=False, refine_psf=False, iter_callback=None):
        """Richardson-Lucy deconvolution.
        Parameters
        ----------
        image : ndarray
           Input degraded image (can be N dimensional).
        psf : ndarray, optional
           The point spread function.
        iterations : int, optional
           Number of iterations. This parameter plays the role of
           regularisation.
        clip : boolean, optional
           True by default. If true, pixel value of the result above 1 or
           under -1 are thresholded for skimage pipeline compatibility.
        return_iterations : boolean, optional
            Returns a list of the PSF and the deconvolved image for each iteration.
        refine_psf : boolean, optional
            Refines the passed PSF by means of blind deconvolution.[2]
            If no PSF is passed, this is automatically turned on.
        Returns
        -------
        im_deconv : ndarray
           The deconvolved image.
        psf : ndarray
            The last PSF estimate to deconvolve image.
        Examples
        --------
        >>> from skimage import color, data, restoration
        >>> camera = color.rgb2gray(data.camera())
        >>> from scipy.signal import convolve2d
        >>> psf = np.ones((5, 5)) / 25
        >>> camera = convolve2d(camera, psf, 'same')
        >>> camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
        >>> deconvolved = restoration.richardson_lucy(camera, psf, 5)
        Notes
        -----
        The Richardson-Lucy algorithm is an iterative algorithm to
        deconvolve an image using a given point spread function (PSF).
        If no PSF is provided, the algorithm performs an "inverse" Richardson
        Lucy algorithm as described in Fish et al., 1995.
        It is an iterative process where the PSF
        and image is deconvolved, respectively.
        It is more noise tolerant than other algorithms,
        such as Ayers-Dainty and the Wiener filter algorithms
        (taken from the paper).
        The algorithm performs well with gaussian PSFs and can recover
        them nicely without any prior knowledge. If one has already an
        educated guess, one should pass the PSF as argument to the function.
        Note, that the PSF should have the same shape as the image,
        and the PSF should be centered.
        Due to its nature, the algorithm may divide by 0.
        The function catches this issue and aborts the iterative process.
        Mostly, the optimal number of iterations is before this error may occur.
        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
        .. [2] Fish, D. A., A. M. Brinicombe, E. R. Pike, and J. G. Walker.
               "Blind deconvolution by means of the Richardson–Lucy algorithm."
               JOSA A 12, no. 1 (1995): 58-65. DOI:`10.1364/JOSAA.12.000058`.
        """

        image = image.astype(np.float)

        # Initialize PSF
        # No PSF was passed, create an ndarray with 0.5 in each cell.
        if psf is None:
            psf = np.full(image.shape, 0.5)
            refine_psf = True  # enable blind deconvolution automatically

        # If PSF was passed and refine_psf option set to true,
        # psf is padded with 0 to allow convolution with image
        # of shape image.shape.
        elif psf is not None and refine_psf:
            psf = self._pad_to_shape(image, psf)
            psf = psf.astype(np.float)

        else:
            psf = psf.astype(np.float)

        # Initialize deconvolution image for estimating PSF
        im_deconv = np.full(image.shape, 0.5)
        psf_mirror = psf[::-1, ::-1]

        # Apply algorithm for finite iterations
        for i in range(iterations):
            # Blind deconvolution or refining PSF if an estimate was passed.
            if refine_psf:
                # Deconvolve the PSF
                # Hack: in original publication one would have used `image`,
                #       however, this does not work.
                #       Using `im_deconv` instead recovers PSF.
                relative_blur_psf = im_deconv / convolve(psf, im_deconv, "same")

                # Check for zeros in PSF, causes the latter code to crash
                if np.count_nonzero(relative_blur_psf == 0):
                    break

                else:
                    psf *= convolve(relative_blur_psf, im_deconv[::-1, ::-1], "same")

                    # Compute inverse again
                    psf_mirror = psf[::-1, ::-1]

            # Perform deconvolution
            relative_blur = image / convolve(im_deconv, psf, "same")
            im_deconv *= convolve(relative_blur, psf_mirror, "same")
            # Add iteration to list, if desired
            if return_iterations:
                iter_callback(im_deconv, psf, i)

        if clip:
            im_deconv[im_deconv > 1] = 1
            im_deconv[im_deconv < -1] = -1

        return im_deconv
