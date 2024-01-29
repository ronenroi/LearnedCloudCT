import numpy as np

class SatelliteNoise(object):
    def __init__(self, fullwell=13.5e3, bits=10, DARK_NOISE_std=13, full_well_val=None):
        GECKO = {'PIXEL_SIZE': 5.5, 'FULLWELL': 13.5e3, 'CHeight': 2048, 'CWidth': 2048,
                 'SENSOR_ID': 0, 'READOUT_NOISE': 13, 'DARK_CURRENT_NOISE': 125, 'TEMP': 25, 'BitDepth': 10}
        self._bits = bits
        self._fullwell = fullwell
        self.DARK_NOISE_std = DARK_NOISE_std
        self._alpha = (2 ** self._bits) / self._fullwell
        self._full_well_val = full_well_val

    def add_noise(self ,electrons_number_image):
        """
        TODO
        Currently, we use:
        * Gaussian distribution for the read noise and dark noise. TODO consider more accurate model for both.


        """
        electrons_number = electrons_number_image.copy()
        # photon noise, by Poisson noise:
        electrons_number = np.random.poisson(electrons_number)

        # dark noise:
        DARK_NOISE_mean = 0 #(self._DARK_NOISE *1e-6 * self._exposure_time)
        DN_noise = np.random.normal(
            loc=DARK_NOISE_mean,
            scale=self.DARK_NOISE_std
            ,  # The scale parameter controls the standard deviation of the normal distribution.
            size=electrons_number.shape
        ).astype(np.int)

        electrons_number += DN_noise

        # read noise:
        # # TODO ask Yoav if it is needed and how to model it?
        # READ_NOISE_mean = (self._READ_NOISE**2)
        # READ_NOISE_variance = (self._READ_NOISE**2) # since it comes from poisson distribution.
        # READ_noise = np.random.normal(
        #     loc=0, # TODO ask Yoav
        #     scale=READ_NOISE_variance**0.5
        #     ,  # The scale parameter controls the standard deviation of the normal distribution.
        #     size=electrons_number.shape
        # ).astype(np.int)
        #
        # electrons_number += READ_noise

        electrons_number = np.clip(electrons_number, a_min = 0, a_max=None)
        return electrons_number

    def convert_radiance_to_graylevel(self, images):
        """
        This method convert radiances to grayscals. In addition, it returns the scale that would convert radiance to grayscale BUT without noise addition in the electrons lavel.
        The user decides if to use that as normalization or not.

        Parameters:
        Input:
        images - np.array or list of np.arrays, it is the images that represent radiances that reache the lens where the simulation of that radiance considered
        solar flux of 1 [W/m^2] and spectrally-dependent parameters of the atmospheric model are spectrally-averaged.

        IF_APPLY_NOISE - bool, if it is True, apply noise.

        IF_SCALE_IDEALLY - bool, if it is True, ignore imager parameters that plays in the convertion of
        radiance to electrons. It may be used just for simulations of ideal senarios or debug.

        Output:
        gray_scales - list of images in grayscale

        radiance_to_graylevel_scale - foalt,
              The scale that would convert radiance to grayscale BUT without noise addition in the electrons lavel
        """

        noisy_images = []
        if self._full_well_val is None or  self._full_well_val == 'None':
            max_of_all_images = np.array(images).max()
        else:
            max_of_all_images = self._full_well_val

        gray_level_bound = 2 ** self._bits


        # Adjust synthetic scale induced by exposure, gain, lens diameter etc. to make maximum signal that reach the full well
        scale = self._fullwell / max_of_all_images
        radiance_to_graylevel_scale = self._alpha * scale


        for index, image in enumerate(images):
            # image - pixels units of [1/st]
            # Adjust synthetic scale induced by exposure, gain, lens diameter etc. to make maximum signal that reach the full well
            electrons_number = image * scale
            electrons_number = self.add_noise(electrons_number)

            # ---------------- finish the noise ------------------------------------------------
            gray_scale = self._alpha * electrons_number

            # For a sensor having a linear radiometric response, the conversion between pixel electrons to grayscale is by a fixxed ratio self._alpha
            # Quantisize and cut overflow values.
            gray_scale = np.round(gray_scale).astype(np.int)
            gray_scale = np.clip(gray_scale, a_min=0, a_max=gray_level_bound)
            noisy_image = gray_scale/radiance_to_graylevel_scale
            noisy_images.append(noisy_image)


        return np.array(noisy_images)

