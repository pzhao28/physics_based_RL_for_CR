import numpy as np

class testIntruderConfig:
    def __init__(self, vels=np.array([4, 4,  2*np.sqrt(5), 2*np.sqrt(5)]),
                        starts = np.array([[80, 0.0], [40, 0], [80, 20], [0, 20.0]])
                        , ends = np.array([[0, 0.0], [40, 80], [40, 0], [40, 0.0]])
                        , steps = np.array([1, 1, 1, 30])):


        self.num = 4
        self.vels = vels
        self.starts = starts
        self.ends = ends
        self.steps = steps

        '''vels=np.array([3.5, 3.8, 3.4, 3.1, 3.7, 4.7, 4.7, 3.6, 3.3, 4.2]),
                        starts = np.array([[49.63715644816473, 80.0], [59.11929139562218, 80.0], [73.29008546876412, 80.0], [31.399024346161035, 80.0], [67.21927957428906, 80.0], [26.814127541824718, 80.0], [34.68607554316485, 80.0], [3.1750013388164344, 80.0], [23.477397528259694, 80.0], [72.32288572536993, 80.0]])
                        , ends = np.array([[22.481032567583945, 0.0], [75.5902842092892, 0.0], [29.016378254004174, 0.0], [73.56723243536041, 0.0], [53.12537374954135, 0.0], [77.93477127798045, 0.0], [59.86732001247393, 0.0], [59.56084980734757, 0.0], [41.90802206624443, 0.0], [53.39231113399278, 0.0]])
                        , steps = np.array([17, 38,  2,  1, 32, 24, 13, 26, 13,  2])'''

        '''vels=np.array([3.5, 3.8]),
               starts = np.array([[20, 80.0], [60, 80.0]])
               , ends = np.array([[60, 0.0], [20, 0.0]])
               , steps = np.array([100, 100])):'''