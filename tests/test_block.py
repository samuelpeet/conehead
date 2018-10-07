import numpy as np
from conehead.block import Block


class TestBlock:
    def test_square_block(self):
        block = Block()
        block.set_square(10)
        correct = np.zeros((400, 400))
        correct[150:250, 150:250] = 1.0
        np.testing.assert_array_almost_equal(correct, block.block_values)

    def test_mlc(self):
        """
         10------ ------1
           -----   -----
           ----     ----
           ---       ---
        A  --         --  B
           --         --
           ---       ---
           ----     ----
           -----   -----
          1------ ------10    
        """

        # position_boundaries = [-50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        # ends = [-5.0, -10.0, -20.0, -30.0, -40.0, -40.0, -30.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 30.0, 40.0, 40.0, 30.0, 20.0, 10.0, 5.0]
        position_boundaries = ['-204.00', '-190.00', '-180.00', '-170.00', '-160.00', '-150.00', '-140.00', '-130.00', '-120.00', '-110.00', '-100.00', '-95.00', '-90.00', '-85.00', '-80.00', '-75.00', '-70.00', '-65.00', '-60.00', '-55.00', '-50.00', '-45.00', '-40.00', '-35.00', '-30.00', '-25.00', '-20.00', '-15.00', '-10.00', '-5.00', '0.00', '5.00', '10.00', '15.00', '20.00', '25.00', '30.00', '35.00', '40.00', '45.00', '50.00', '55.00', '60.00', '65.00', '70.00', '75.00', '80.00', '85.00', '90.00', '95.00', '100.00', '110.00', '120.00', '130.00', '140.00', '150.00', '160.00', '170.00', '180.00', '190.00', '204.00']
        # ends = ['0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '7.04', '-27.34', '-27.46', '-27.19', '-26.84', '-34.08', '-33.61', '-34.12', '-41.83', '-41.83', '-42.25', '-47.96', '-51.11', '-52.49', '-52.72', '-52.42', '-47.58', '-46.38', '-40.11', '-38.05', '-30.56', '-28.84', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '20.52', '20.58', '27.08', '33.74', '33.42', '38.57', '35.03', '40.62', '41.11', '47.75', '48.69', '48.39', '48.46', '47.90', '47.02', '47.24', '47.63', '41.19', '41.78', '35.30', '19.88', '-1.11', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00']
        ends = ['0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '-4.01', '-20.58', '-26.97', '-26.24', '-33.27', '-40.47', '-40.89', '-41.31', '-40.68', '-40.99', '-41.40', '-47.99', '-48.14', '-47.86', '-47.46', '-37.95', '-38.79', '-41.15', '-16.21', '4.53', '5.02', '4.21', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '-3.01', '20.17', '20.20', '27.02', '27.16', '33.85', '33.52', '41.74', '48.20', '48.21', '39.41', '24.43', '13.40', '15.63', '15.93', '13.53', '12.17', '4.74', '11.54', '8.21', '8.31', '5.21', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00']


        # Convert to tenths of a millimetre
        position_boundaries = np.array(position_boundaries).astype(np.float)
        position_boundaries = np.floor(position_boundaries * 10)
        # Convert to tenths of a millimetre
        ends = np.array(ends).astype(np.float)
        ends = np.floor(np.array(ends) * 10)

        # Total width of MLC bank
        mlc_width = int(np.abs(position_boundaries[0] - position_boundaries[-1]))
        
        class Leaf:
            def __init__(self, min_bound, max_bound, end, bank):
                self.min_bound = min_bound
                self.max_bound = max_bound
                self.width = int(np.abs(max_bound - min_bound))
                self.end = int(end)
                self.bank = bank

                self.r_min = int(min_bound + np.abs(position_boundaries[0]))
                self.r_max = int(self.r_min + self.width)

                if bank == 'A':
                    self.c_min = 0
                    self.c_max = int(2000 + end)
                    self.area = self._leaf_transmission(self.width, self.c_max, bank)

                elif bank == 'B':
                    self.c_min = int(2000 + end)
                    self.c_max = 3999
                    self.area = self._leaf_transmission(self.width, self.c_max - self.c_min, bank)
                else:
                    assert False, \
                    "bank must be \'A\' or \'B\'"                  

            def _leaf_transmission(self, width, height, bank):
                    area = np.ones((width, height))
                    area[0, :] = 0.20
                    area[1, :] = 0.50
                    area[2, :] = 0.75
                    area[-1, :] = 0.20
                    area[-2, :] = 0.50
                    area[-3, :] = 0.75
                    area[:, -15:] *= np.linspace(1.0, 0.0, 15)
                    if bank == 'B':
                        area = np.fliplr(np.flipud(area))
                    return area

        ends_a = ends[:int(len(ends)/2)]
        ends_b = ends[int(len(ends)/2):]

        leaves_a = []
        for n, _ in enumerate(position_boundaries):
            if n == len(position_boundaries) - 1:
                break
            else:
                leaf = Leaf(
                    position_boundaries[n], position_boundaries[n+1], ends_a[n], 'A' 
                )
                leaves_a.append(leaf)

        leaves_b = []
        for n, _ in enumerate(position_boundaries):
            if n == len(position_boundaries) - 1:
                break
            leaf = Leaf(
                position_boundaries[n], position_boundaries[n+1], ends_b[n], 'B' 
            )
            leaves_b.append(leaf)

        xmin, xmax, xnum = (position_boundaries[0], position_boundaries[-1], mlc_width)
        xres = xnum / (xmax - xmin)
        ymin, ymax, ynum = (-20, 20, 4000)
        yres = ynum / (ymax - ymin)
        block_locations = np.mgrid[
                xmin:xmax:xnum*1j,
                ymin:ymax:ynum*1j
            ]
        block_values = np.zeros((xnum, ynum))

        # print(leaves_a[0].min_bound, position_boundaries[0])
        for n, l in enumerate(leaves_a):
            
            try:    
                block_values[l.r_min:l.r_max, l.c_min:l.c_max] += l.area
            except ValueError:
                print(n, l.r_min, l.r_max, l.c_min, l.c_max, l.end, l.area.shape)
                assert False

        for l in leaves_b:
            print(l.r_min, l.r_max, l.c_max)
            block_values[l.r_min:l.r_max, l.c_min:l.c_max] += l.area



        # import matplotlib.pyplot as plt
        # plt.imshow(block_values)
        # plt.show()
